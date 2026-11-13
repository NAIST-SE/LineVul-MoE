from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data import WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup

from common import (
    TOP_P_GRID,
    binary_metrics,
    effective_number_weights,
    fit_binary_temperature,
    fit_multiclass_temperature,
    macro_cwe_average_precision,
    noisy_or_fusion,
    read_json,
    set_seed,
    sigmoid,
    softmax,
    stable_unit_interval,
    threshold_for_best_f1,
    top_p_mask,
    write_json,
)
from models import build_model, load_warm_start, parameter_counts


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        cache_dir: Path,
        split: str,
        seed: int,
        positive_fraction: float,
        model_kind: str,
        expert_index: int | None,
        expert_negative_mode: str,
        max_rows: int | None,
    ):
        frame = pd.read_csv(manifest_path, low_memory=False)
        frame = frame[frame["split"] == split].copy()
        if split == "train" and positive_fraction < 1.0:
            top5_positive = (frame["target"] == 1) & (frame["cwe_index"] >= 0)
            keep_positive = np.array(
                [
                    stable_unit_interval(seed, row_id) < positive_fraction
                    for row_id in frame.loc[top5_positive, "row_id"]
                ],
                dtype=bool,
            )
            retained = np.ones(len(frame), dtype=bool)
            retained[np.flatnonzero(top5_positive.to_numpy())] = keep_positive
            frame = frame.iloc[np.flatnonzero(retained)].copy()

        if split == "train" and model_kind == "specialist":
            if expert_index is None:
                raise ValueError("specialist requires --expert-index")
            frame = frame[
                specialist_population_mask(
                    frame["target"].to_numpy(),
                    frame["cwe_index"].to_numpy(),
                    expert_index,
                    expert_negative_mode,
                )
            ].copy()

        if max_rows is not None and len(frame) > max_rows:
            frame["_selection"] = [
                stable_unit_interval(seed + 991, row_id)
                for row_id in frame["row_id"]
            ]
            strata = list(frame["stratum"].drop_duplicates())
            per_stratum = max(max_rows // max(len(strata), 1), 1)
            selected_indices: list[int] = []
            for _, group in frame.groupby("stratum", sort=False):
                selected_indices.extend(
                    group.nsmallest(min(len(group), per_stratum), "_selection").index
                )
            if len(selected_indices) < max_rows:
                remaining = frame.drop(index=selected_indices)
                selected_indices.extend(
                    remaining.nsmallest(
                        min(max_rows - len(selected_indices), len(remaining)),
                        "_selection",
                    ).index
                )
            frame = frame.loc[selected_indices].drop(columns="_selection")

        self.frame = frame.reset_index(drop=True)
        self.cache: dict[int, np.ndarray] = {}
        for source_code, source in enumerate(("train", "valid", "test")):
            path = cache_dir / f"{source}.input_ids.npy"
            if not path.is_file():
                raise FileNotFoundError(path)
            self.cache[source_code] = np.load(path, mmap_mode="r")
        self.source_codes = self.frame["source_code"].to_numpy(dtype=np.int8)
        self.source_indices = self.frame["source_index"].to_numpy(dtype=np.int64)
        self.targets = self.frame["target"].to_numpy(dtype=np.int64)
        self.cwe_indices = self.frame["cwe_index"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source_code = int(self.source_codes[index])
        source_index = int(self.source_indices[index])
        input_ids = np.array(
            self.cache[source_code][source_index],
            dtype=np.int64,
            copy=True,
        )
        return (
            torch.from_numpy(input_ids),
            torch.tensor(self.targets[index], dtype=torch.long),
            torch.tensor(self.cwe_indices[index], dtype=torch.long),
            torch.tensor(index, dtype=torch.long),
        )


def class_id(target: torch.Tensor, cwe_index: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(target)
    top5 = (target == 1) & (cwe_index >= 0)
    result[top5] = cwe_index[top5] + 1
    result[(target == 1) & (cwe_index < 0)] = 6
    return result


def specialist_population_mask(
    target: np.ndarray,
    cwe_index: np.ndarray,
    expert_index: int,
    expert_negative_mode: str,
) -> np.ndarray:
    """Select the population on which a specialist is trained/evaluated.

    ``conditional-one-hot`` estimates P(CWE=k | vulnerable, CWE in Top5):
    the target CWE is positive and the other four Top5 CWEs are negative.
    Clean, tail, and unknown examples are deliberately outside that task.
    """
    if expert_negative_mode == "exclude":
        return (target == 0) | ((target == 1) & (cwe_index == expert_index))
    if expert_negative_mode == "one-vs-rest":
        return np.ones_like(target, dtype=bool)
    if expert_negative_mode == "conditional-one-hot":
        return (target == 1) & (cwe_index >= 0) & (cwe_index < 5)
    raise ValueError(expert_negative_mode)


def make_loader(
    dataset: ManifestDataset,
    batch_size: int,
    workers: int,
    train: bool,
    imbalance: str,
    seed: int,
) -> DataLoader:
    if train and imbalance == "oversample":
        strata = dataset.frame["stratum"].astype(str)
        counts = strata.value_counts()
        weights = strata.map(lambda value: 1.0 / counts[value]).to_numpy()
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    elif train:
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def loss_context(
    dataset: ManifestDataset,
    model_kind: str,
    expert_index: int | None,
) -> dict[str, Any]:
    targets = dataset.targets
    cwe_indices = dataset.cwe_indices
    context: dict[str, Any] = {
        "binary_weights": effective_number_weights(
            [int((targets == 0).sum()), int((targets == 1).sum())]
        )
    }
    classes = np.zeros_like(targets)
    top5 = (targets == 1) & (cwe_indices >= 0)
    classes[top5] = cwe_indices[top5] + 1
    classes[(targets == 1) & (cwe_indices < 0)] = 6
    context["multiclass_weights"] = effective_number_weights(
        [int((classes == index).sum()) for index in range(7)]
    )
    router = np.where(cwe_indices >= 0, cwe_indices, 5)
    vulnerable = targets == 1
    context["router_weights"] = effective_number_weights(
        [int((router[vulnerable] == index).sum()) for index in range(6)]
    )
    context["expert_weights"] = []
    for index in range(5):
        mask = (targets == 0) | ((targets == 1) & (cwe_indices == index))
        labels = ((targets == 1) & (cwe_indices == index))[mask]
        context["expert_weights"].append(
            effective_number_weights(
                [int((labels == 0).sum()), int((labels == 1).sum())]
            )
        )
    if model_kind == "specialist" and expert_index is not None:
        labels = (
            (targets == 1) & (cwe_indices == expert_index)
        ).astype(np.int64)
        context["specialist_weights"] = effective_number_weights(
            [int((labels == 0).sum()), int((labels == 1).sum())]
        )
    return context


def weighted_binary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: np.ndarray | None,
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(
        logits,
        labels.float(),
        reduction="none",
    )
    if weights is not None:
        tensor_weights = torch.as_tensor(
            weights,
            dtype=losses.dtype,
            device=losses.device,
        )
        losses = losses * tensor_weights[labels.long()]
    return losses.mean()


def compute_loss(
    model_kind: str,
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    cwe_indices: torch.Tensor,
    imbalance: str,
    context: dict[str, Any],
    expert_index: int | None,
    expert_negative_mode: str,
    router_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    use_weights = imbalance == "class_balanced"
    if model_kind == "generalist":
        logits = outputs["logits"].squeeze(-1)
        loss = weighted_binary_loss(
            logits,
            targets,
            context["binary_weights"] if use_weights else None,
        )
        return loss, {"generalist": float(loss.detach())}

    if model_kind == "multiclass":
        labels = class_id(targets, cwe_indices)
        weights = (
            torch.as_tensor(
                context["multiclass_weights"],
                dtype=outputs["logits"].dtype,
                device=outputs["logits"].device,
            )
            if use_weights
            else None
        )
        loss = F.cross_entropy(outputs["logits"], labels, weight=weights)
        return loss, {"multiclass": float(loss.detach())}

    if model_kind == "specialist":
        if expert_index is None:
            raise ValueError("specialist requires expert index")
        labels = (
            (targets == 1) & (cwe_indices == expert_index)
        ).long()
        loss = weighted_binary_loss(
            outputs["logits"].squeeze(-1),
            labels,
            context["specialist_weights"] if use_weights else None,
        )
        return loss, {"specialist": float(loss.detach())}

    generalist_loss = weighted_binary_loss(
        outputs["generalist_logits"],
        targets,
        context["binary_weights"] if use_weights else None,
    )
    expert_losses: list[torch.Tensor] = []
    for index in range(5):
        labels = ((targets == 1) & (cwe_indices == index)).long()
        if expert_negative_mode == "exclude":
            mask = (targets == 0) | labels.bool()
        else:
            mask = torch.ones_like(targets, dtype=torch.bool)
        if mask.any():
            expert_losses.append(
                weighted_binary_loss(
                    outputs["expert_logits"][mask, index],
                    labels[mask],
                    context["expert_weights"][index] if use_weights else None,
                )
            )
    expert_loss = torch.stack(expert_losses).mean()
    vulnerable = targets == 1
    if vulnerable.any():
        router_labels = torch.where(
            cwe_indices[vulnerable] >= 0,
            cwe_indices[vulnerable],
            torch.full_like(cwe_indices[vulnerable], 5),
        )
        router_weights = (
            torch.as_tensor(
                context["router_weights"],
                dtype=outputs["router_logits"].dtype,
                device=outputs["router_logits"].device,
            )
            if use_weights
            else None
        )
        router_loss = F.cross_entropy(
            outputs["router_logits"][vulnerable],
            router_labels,
            weight=router_weights,
        )
    else:
        router_loss = outputs["router_logits"].sum() * 0.0
    total = generalist_loss + expert_loss + router_loss_weight * router_loss
    return total, {
        "generalist": float(generalist_loss.detach()),
        "experts": float(expert_loss.detach()),
        "router": float(router_loss.detach()),
    }


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    model_kind: str,
    device: torch.device,
    pad_token_id: int,
) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        "target": [],
        "cwe_index": [],
        "dataset_index": [],
    }
    if model_kind == "sme":
        collected.update(
            {
                "generalist_logits": [],
                "expert_logits": [],
                "router_logits": [],
            }
        )
    else:
        collected["logits"] = []

    for input_ids, targets, cwe_indices, dataset_indices in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = input_ids.ne(pad_token_id)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        collected["target"].append(targets.numpy())
        collected["cwe_index"].append(cwe_indices.numpy())
        collected["dataset_index"].append(dataset_indices.numpy())
        if model_kind == "sme":
            for key in ("generalist_logits", "expert_logits", "router_logits"):
                collected[key].append(outputs[key].detach().cpu().numpy())
        else:
            collected["logits"].append(outputs["logits"].detach().cpu().numpy())
    return {
        key: np.concatenate(values, axis=0)
        for key, values in collected.items()
    }


def validation_score(
    prediction: dict[str, np.ndarray],
    model_kind: str,
    expert_index: int | None,
    expert_negative_mode: str,
) -> float:
    target = prediction["target"]
    cwe_index = prediction["cwe_index"]
    if model_kind == "generalist":
        probability = sigmoid(prediction["logits"].squeeze(-1))
        score, _ = macro_cwe_average_precision(
            target,
            cwe_index,
            probability,
        )
        return score
    if model_kind == "multiclass":
        probability = softmax(prediction["logits"], axis=1)[:, 1:6]
        score, _ = macro_cwe_average_precision(
            target,
            cwe_index,
            probability,
        )
        return score
    if model_kind == "specialist":
        if expert_index is None:
            raise ValueError("specialist requires expert index")
        selected = specialist_population_mask(
            target,
            cwe_index,
            expert_index,
            expert_negative_mode,
        )
        labels = (
            (target[selected] == 1) & (cwe_index[selected] == expert_index)
        ).astype(np.int64)
        return float(
            average_precision_score(
                labels,
                sigmoid(prediction["logits"].squeeze(-1)[selected]),
            )
        )
    general = sigmoid(prediction["generalist_logits"])
    experts = sigmoid(prediction["expert_logits"])
    router = softmax(prediction["router_logits"], axis=1)
    scores: list[float] = []
    for top_p in TOP_P_GRID:
        selected = top_p_mask(router, top_p, max_experts=3)[:, :5]
        per_cwe = 1.0 - (1.0 - general[:, None]) * (
            1.0
            - np.where(
                selected,
                router[:, :5] * experts,
                0.0,
            )
        )
        score, _ = macro_cwe_average_precision(
            target,
            cwe_index,
            per_cwe,
        )
        scores.append(score)
    return float(max(scores))


def calibrate_and_fuse(
    validation: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    model_kind: str,
    expert_index: int | None,
    expert_negative_mode: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    validation_target = validation["target"]
    validation_cwe = validation["cwe_index"]

    if model_kind in {"generalist", "specialist"}:
        validation_logits = validation["logits"].squeeze(-1)
        test_logits = test["logits"].squeeze(-1)
        if model_kind == "specialist":
            if expert_index is None:
                raise ValueError("specialist requires expert index")
            calibration_mask = specialist_population_mask(
                validation_target,
                validation_cwe,
                expert_index,
                expert_negative_mode,
            )
            calibration_labels = (
                (validation_target[calibration_mask] == 1)
                & (validation_cwe[calibration_mask] == expert_index)
            ).astype(np.int64)
            temperature = fit_binary_temperature(
                validation_logits[calibration_mask],
                calibration_labels,
            )
            validation_probability = sigmoid(validation_logits / temperature)
            threshold = threshold_for_best_f1(
                calibration_labels,
                validation_probability[calibration_mask],
            )
        else:
            temperature = fit_binary_temperature(
                validation_logits,
                validation_target,
            )
            validation_probability = sigmoid(validation_logits / temperature)
            threshold = threshold_for_best_f1(
                validation_target,
                validation_probability,
            )
        test_probability = sigmoid(test_logits / temperature)
        calibration = {
            "temperature": temperature,
            "threshold": threshold,
        }
        return (
            calibration,
            {
                "final_probability": validation_probability,
                "per_cwe_probability": np.repeat(
                    validation_probability[:, None],
                    5,
                    axis=1,
                ),
            },
            {
                "final_probability": test_probability,
                "per_cwe_probability": np.repeat(
                    test_probability[:, None],
                    5,
                    axis=1,
                ),
            },
        )

    if model_kind == "multiclass":
        validation_labels = np.zeros_like(validation_target)
        top5 = (validation_target == 1) & (validation_cwe >= 0)
        validation_labels[top5] = validation_cwe[top5] + 1
        validation_labels[
            (validation_target == 1) & (validation_cwe < 0)
        ] = 6
        temperature = fit_multiclass_temperature(
            validation["logits"],
            validation_labels,
        )
        validation_probability = softmax(
            validation["logits"] / temperature,
            axis=1,
        )
        test_probability = softmax(test["logits"] / temperature, axis=1)
        validation_final = 1.0 - validation_probability[:, 0]
        test_final = 1.0 - test_probability[:, 0]
        threshold = threshold_for_best_f1(
            validation_target,
            validation_final,
        )
        calibration = {
            "temperature": temperature,
            "threshold": threshold,
        }
        return (
            calibration,
            {
                "final_probability": validation_final,
                "per_cwe_probability": validation_probability[:, 1:6],
            },
            {
                "final_probability": test_final,
                "per_cwe_probability": test_probability[:, 1:6],
            },
        )

    general_temperature = fit_binary_temperature(
        validation["generalist_logits"],
        validation_target,
    )
    expert_temperatures: list[float] = []
    for index in range(5):
        if expert_negative_mode == "exclude":
            mask = (validation_target == 0) | (
                (validation_target == 1) & (validation_cwe == index)
            )
        else:
            mask = np.ones_like(validation_target, dtype=bool)
        labels = (
            (validation_target[mask] == 1) & (validation_cwe[mask] == index)
        ).astype(np.int64)
        expert_temperatures.append(
            fit_binary_temperature(
                validation["expert_logits"][mask, index],
                labels,
            )
        )
    vulnerable = validation_target == 1
    router_labels = np.where(
        validation_cwe[vulnerable] >= 0,
        validation_cwe[vulnerable],
        5,
    )
    router_temperature = fit_multiclass_temperature(
        validation["router_logits"][vulnerable],
        router_labels,
    )

    def probabilities(
        raw: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        general = sigmoid(raw["generalist_logits"] / general_temperature)
        experts = sigmoid(
            raw["expert_logits"]
            / np.asarray(expert_temperatures, dtype=np.float64)[None, :]
        )
        router = softmax(
            raw["router_logits"] / router_temperature,
            axis=1,
        )
        return general, experts, router

    validation_general, validation_experts, validation_router = probabilities(
        validation
    )
    test_general, test_experts, test_router = probabilities(test)
    selection_scores: dict[str, float] = {}
    for top_p in TOP_P_GRID:
        selected = top_p_mask(validation_router, top_p, max_experts=3)
        selected_top5 = selected[:, :5]
        routed_per_cwe = 1.0 - (
            1.0 - validation_general[:, None]
        ) * (
            1.0
            - np.where(
                selected_top5,
                validation_router[:, :5] * validation_experts,
                0.0,
            )
        )
        score, _ = macro_cwe_average_precision(
            validation_target,
            validation_cwe,
            routed_per_cwe,
        )
        selection_scores[str(top_p)] = score
    selected_top_p = max(
        TOP_P_GRID,
        key=lambda value: (selection_scores[str(value)], -value),
    )

    def fuse(
        general: np.ndarray,
        experts: np.ndarray,
        router: np.ndarray,
    ) -> dict[str, np.ndarray]:
        selected_all = top_p_mask(router, selected_top_p, max_experts=3)
        selected_top5 = selected_all[:, :5]
        final = noisy_or_fusion(
            general,
            experts,
            router[:, :5],
            selected_top5,
        )
        per_cwe = 1.0 - (1.0 - general[:, None]) * (
            1.0
            - np.where(
                selected_top5,
                router[:, :5] * experts,
                0.0,
            )
        )
        return {
            "final_probability": final,
            "per_cwe_probability": per_cwe,
            "general_probability": general,
            "expert_probability": experts,
            "router_probability": router,
            "selected_mask": selected_all,
        }

    validation_fused = fuse(
        validation_general,
        validation_experts,
        validation_router,
    )
    test_fused = fuse(test_general, test_experts, test_router)
    threshold = threshold_for_best_f1(
        validation_target,
        validation_fused["final_probability"],
    )
    calibration = {
        "general_temperature": general_temperature,
        "expert_temperatures": expert_temperatures,
        "router_temperature": router_temperature,
        "top_p": selected_top_p,
        "top_p_validation_macro_ap": selection_scores,
        "threshold": threshold,
    }
    return calibration, validation_fused, test_fused


def evaluate_outputs(
    prediction: dict[str, np.ndarray],
    fused: dict[str, np.ndarray],
    threshold: float,
    top5: list[str],
    cwe_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    target = prediction["target"]
    cwe_index = prediction["cwe_index"]
    final = fused["final_probability"]
    per_cwe = fused["per_cwe_probability"]
    macro_ap, per_cwe_ap = macro_cwe_average_precision(
        target,
        cwe_index,
        per_cwe,
    )
    metrics: dict[str, Any] = {
        "system": binary_metrics(target, final, threshold),
        "top5_macro_average_precision": macro_ap,
        "per_cwe": {},
    }
    for index, cwe in enumerate(top5):
        selected = (target == 0) | (
            (target == 1) & (cwe_index == index)
        )
        labels = (
            (target[selected] == 1) & (cwe_index[selected] == index)
        ).astype(np.int64)
        probabilities = per_cwe[selected, index]
        cwe_threshold = (
            float(cwe_thresholds[index])
            if cwe_thresholds is not None
            else float(threshold)
        )
        predictions = (probabilities >= cwe_threshold).astype(np.int64)
        metrics["per_cwe"][cwe] = {
            "average_precision": per_cwe_ap[str(index)],
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "positives": int(labels.sum()),
            "threshold": cwe_threshold,
        }
    system_predictions = final >= threshold
    strata = {
        "top5": (target == 1) & (cwe_index >= 0),
        "tail": (target == 1) & (cwe_index == -1),
        "unknown": (target == 1) & (cwe_index == -2),
    }
    metrics["stratum_recall"] = {
        name: (
            float(system_predictions[mask].mean())
            if mask.any()
            else float("nan")
        )
        for name, mask in strata.items()
    }
    finite_recalls = [
        value["recall"]
        for value in metrics["per_cwe"].values()
        if np.isfinite(value["recall"])
    ]
    metrics["worst_cwe_recall"] = (
        float(min(finite_recalls)) if finite_recalls else float("nan")
    )
    if "router_probability" in fused:
        vulnerable = target == 1
        router_labels = np.where(cwe_index[vulnerable] >= 0, cwe_index[vulnerable], 5)
        router = fused["router_probability"][vulnerable]
        order = np.argsort(-router, axis=1)
        metrics["router"] = {
            "top1_recall": float((order[:, 0] == router_labels).mean()),
            "top3_recall": float(
                np.any(order[:, :3] == router_labels[:, None], axis=1).mean()
            ),
            "routing_induced_false_positives": int(
                (
                    (target == 0)
                    & system_predictions
                    & (fused["general_probability"] < threshold)
                ).sum()
            ),
            "clean_false_positives_routed_to_top5": int(
                (
                    (target == 0)
                    & system_predictions
                    & (np.argmax(fused["router_probability"], axis=1) < 5)
                ).sum()
            ),
        }
    return metrics


def validation_cwe_thresholds(
    target: np.ndarray,
    cwe_index: np.ndarray,
    per_cwe_probability: np.ndarray,
) -> list[float]:
    thresholds: list[float] = []
    for index in range(5):
        selected = (target == 0) | ((target == 1) & (cwe_index == index))
        labels = (
            (target[selected] == 1) & (cwe_index[selected] == index)
        ).astype(np.int64)
        thresholds.append(
            threshold_for_best_f1(
                labels,
                per_cwe_probability[selected, index],
            )
        )
    return thresholds


def prediction_frame(
    dataset: ManifestDataset,
    prediction: dict[str, np.ndarray],
    fused: dict[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    order = prediction["dataset_index"].astype(np.int64)
    frame = dataset.frame.iloc[order].copy().reset_index(drop=True)
    frame["probability"] = fused["final_probability"]
    frame["prediction"] = (
        fused["final_probability"] >= threshold
    ).astype(np.int8)
    for index in range(5):
        frame[f"cwe_probability_{index}"] = fused["per_cwe_probability"][:, index]
    if "general_probability" in fused:
        frame["general_probability"] = fused["general_probability"]
        for index in range(5):
            frame[f"expert_probability_{index}"] = fused[
                "expert_probability"
            ][:, index]
        for index in range(6):
            frame[f"router_probability_{index}"] = fused[
                "router_probability"
            ][:, index]
        frame["selected_experts"] = [
            ",".join(str(index) for index in np.flatnonzero(row))
            for row in fused["selected_mask"]
        ]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-kind",
        choices=("generalist", "multiclass", "specialist", "sme"),
        required=True,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--top5", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--expert-index", type=int, choices=range(5))
    parser.add_argument(
        "--expert-negative-mode",
        choices=("exclude", "one-vs-rest", "conditional-one-hot"),
        default="exclude",
    )
    parser.add_argument(
        "--imbalance",
        choices=("bce", "oversample", "class_balanced"),
        default="bce",
    )
    parser.add_argument("--positive-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--adapter-size", type=int, default=64)
    parser.add_argument("--router-loss-weight", type=float, default=0.2)
    parser.add_argument("--pad-token-id", type=int, default=1)
    parser.add_argument(
        "--precision",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-eval-rows", type=int)
    parser.add_argument("--max-test-rows", type=int)
    args = parser.parse_args()

    if not 0.0 < args.positive_fraction <= 1.0:
        raise ValueError("--positive-fraction must be in (0, 1]")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    top5 = read_json(args.top5)
    if not isinstance(top5, list) or len(top5) != 5:
        raise ValueError("top5.json must contain five CWE strings")

    train_dataset = ManifestDataset(
        args.manifest,
        args.cache_dir,
        "train",
        args.seed,
        args.positive_fraction,
        args.model_kind,
        args.expert_index,
        args.expert_negative_mode,
        args.max_train_rows,
    )
    validation_dataset = ManifestDataset(
        args.manifest,
        args.cache_dir,
        "valid",
        args.seed,
        1.0,
        args.model_kind,
        args.expert_index,
        args.expert_negative_mode,
        args.max_eval_rows,
    )
    test_dataset = ManifestDataset(
        args.manifest,
        args.cache_dir,
        "test",
        args.seed,
        1.0,
        args.model_kind,
        args.expert_index,
        args.expert_negative_mode,
        args.max_test_rows,
    )
    train_loader = make_loader(
        train_dataset,
        args.train_batch_size,
        args.workers,
        train=True,
        imbalance=args.imbalance,
        seed=args.seed,
    )
    validation_loader = make_loader(
        validation_dataset,
        args.eval_batch_size,
        args.workers,
        train=False,
        imbalance="bce",
        seed=args.seed,
    )
    test_loader = make_loader(
        test_dataset,
        args.eval_batch_size,
        args.workers,
        train=False,
        imbalance="bce",
        seed=args.seed,
    )

    model = build_model(args.model_kind, args.model_path, args.adapter_size)
    warm_start_report = None
    if args.warm_start is not None:
        warm_start_report = load_warm_start(model, args.warm_start)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    counts = parameter_counts(model)
    context = loss_context(
        train_dataset,
        args.model_kind,
        args.expert_index,
    )

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    optimizer = AdamW(
        [
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not any(value in name for value in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if any(value in name for value in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=args.learning_rate,
    )
    updates_per_epoch = int(
        np.ceil(len(train_loader) / args.gradient_accumulation_steps)
    )
    total_updates = max(updates_per_epoch * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    amp_enabled = device.type == "cuda" and args.precision != "fp32"
    amp_dtype = (
        torch.bfloat16 if args.precision == "bf16" else torch.float16
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled and args.precision == "fp16"
    )
    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_checkpoint = args.output_dir / "best.pt"
    start_time = time.monotonic()
    peak_memory = 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        steps = 0
        component_totals: dict[str, float] = {}
        for batch_index, (
            input_ids,
            targets,
            cwe_indices,
            _,
        ) in enumerate(train_loader):
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            cwe_indices = cwe_indices.to(device, non_blocking=True)
            attention_mask = input_ids.ne(args.pad_token_id)
            with torch.cuda.amp.autocast(
                enabled=amp_enabled,
                dtype=amp_dtype,
            ):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss, components = compute_loss(
                    args.model_kind,
                    outputs,
                    targets,
                    cwe_indices,
                    args.imbalance,
                    context,
                    args.expert_index,
                    args.expert_negative_mode,
                    args.router_loss_weight,
                )
                scaled_loss = loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            should_step = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += float(loss.detach())
            steps += 1
            for name, value in components.items():
                component_totals[name] = component_totals.get(name, 0.0) + value
            if device.type == "cuda":
                peak_memory = max(
                    peak_memory,
                    torch.cuda.max_memory_allocated(device),
                )

        validation_raw = predict(
            model,
            validation_loader,
            args.model_kind,
            device,
            args.pad_token_id,
        )
        score = validation_score(
            validation_raw,
            args.model_kind,
            args.expert_index,
            args.expert_negative_mode,
        )
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss / max(steps, 1),
            "components": {
                name: value / max(steps, 1)
                for name, value in component_totals.items()
            },
            "validation_score": score,
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_seconds": time.monotonic() - start_time,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, sort_keys=True), flush=True)
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "top5": top5,
                    "validation_score": score,
                    "epoch": epoch + 1,
                },
                best_checkpoint,
            )
            write_json(args.output_dir / "history.json", history)

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    validation_raw = predict(
        model,
        validation_loader,
        args.model_kind,
        device,
        args.pad_token_id,
    )
    test_raw = predict(
        model,
        test_loader,
        args.model_kind,
        device,
        args.pad_token_id,
    )
    calibration, validation_fused, test_fused = calibrate_and_fuse(
        validation_raw,
        test_raw,
        args.model_kind,
        args.expert_index,
        args.expert_negative_mode,
    )
    cwe_thresholds = validation_cwe_thresholds(
        validation_raw["target"],
        validation_raw["cwe_index"],
        validation_fused["per_cwe_probability"],
    )
    calibration["cwe_thresholds"] = cwe_thresholds
    test_metrics = evaluate_outputs(
        test_raw,
        test_fused,
        float(calibration["threshold"]),
        top5,
        cwe_thresholds,
    )
    validation_metrics = evaluate_outputs(
        validation_raw,
        validation_fused,
        float(calibration["threshold"]),
        top5,
        cwe_thresholds,
    )
    predictions = prediction_frame(
        test_dataset,
        test_raw,
        test_fused,
        float(calibration["threshold"]),
    )
    validation_predictions = prediction_frame(
        validation_dataset,
        validation_raw,
        validation_fused,
        float(calibration["threshold"]),
    )
    predictions.to_csv(
        args.output_dir / "predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    validation_predictions.to_csv(
        args.output_dir / "validation_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    total_seconds = time.monotonic() - start_time
    routing_compute = None
    if args.model_kind == "sme":
        expert_parameters = sum(
            parameter.numel()
            for parameter in model.experts[0].parameters()
        )
        mean_selected_top5 = float(
            test_fused["selected_mask"][:, :5].sum(axis=1).mean()
        )
        routing_compute = {
            "mean_selected_top5_experts": mean_selected_top5,
            "parameters_per_expert": expert_parameters,
            "estimated_routed_active_parameters": int(
                counts["total"]
                - expert_parameters * len(model.experts)
                + expert_parameters * mean_selected_top5
            ),
            "evaluation_computed_all_experts_for_ablation": True,
        }
    run = {
        "model_kind": args.model_kind,
        "expert_index": args.expert_index,
        "expert_negative_mode": args.expert_negative_mode,
        "imbalance": args.imbalance,
        "positive_fraction": args.positive_fraction,
        "precision": args.precision,
        "seed": args.seed,
        "top5": top5,
        "train_rows": len(train_dataset),
        "validation_rows": len(validation_dataset),
        "test_rows": len(test_dataset),
        "best_epoch": checkpoint["epoch"],
        "best_validation_score": checkpoint["validation_score"],
        "calibration": calibration,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "parameter_counts": counts,
        "routing_compute": routing_compute,
        "peak_gpu_memory_bytes": int(peak_memory),
        "total_seconds": total_seconds,
        "examples_per_second": float(
            (len(train_dataset) * args.epochs + len(validation_dataset) * args.epochs)
            / max(total_seconds, 1e-12)
        ),
        "hostname": os.uname().nodename,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "warm_start": warm_start_report,
        "history": history,
    }
    write_json(args.output_dir / "run.json", run)
    print(json.dumps({"event": "completed", **run}, default=str), flush=True)


if __name__ == "__main__":
    main()
