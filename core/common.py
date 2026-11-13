from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
)


CWE_PATTERN = re.compile(r"CWE-\d+")
SOURCE_CODES = {"train": 0, "valid": 1, "test": 2}
SOURCE_NAMES = {value: key for key, value in SOURCE_CODES.items()}
TOP_P_GRID = (0.6, 0.8, 0.9)
TEMPERATURE_GRID = np.logspace(-1.0, 1.0, 81)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_cwe(value: object) -> str:
    if value is None:
        return "unknown"
    match = CWE_PATTERN.search(str(value))
    return match.group(0) if match else "unknown"


def stable_hash(text: object) -> str:
    return hashlib.sha256(str(text).encode("utf-8", errors="replace")).hexdigest()


def stable_unit_interval(seed: int, key: object) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") / float(2**64)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    positive = logits >= 0
    result = np.empty_like(logits, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    result[~positive] = exp_logits / (1.0 + exp_logits)
    return result


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=axis, keepdims=True)


def binary_log_loss_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
) -> float:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    labels = np.asarray(labels, dtype=np.float64)
    losses = np.maximum(scaled, 0.0) - scaled * labels + np.log1p(
        np.exp(-np.abs(scaled))
    )
    return float(np.mean(losses))


def fit_binary_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 1.0
    losses = [
        binary_log_loss_from_logits(logits, labels, float(temperature))
        for temperature in TEMPERATURE_GRID
    ]
    return float(TEMPERATURE_GRID[int(np.argmin(losses))])


def multiclass_log_loss_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
) -> float:
    probabilities = np.clip(
        softmax(np.asarray(logits) / float(temperature), axis=1),
        1e-12,
        1.0,
    )
    return float(-np.log(probabilities[np.arange(len(labels)), labels]).mean())


def fit_multiclass_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
) -> float:
    if len(np.unique(labels)) < 2:
        return 1.0
    losses = [
        multiclass_log_loss_from_logits(logits, labels, float(temperature))
        for temperature in TEMPERATURE_GRID
    ]
    return float(TEMPERATURE_GRID[int(np.argmin(losses))])


def top_p_mask(
    router_probabilities: np.ndarray,
    cumulative_probability: float,
    max_experts: int = 3,
) -> np.ndarray:
    probabilities = np.asarray(router_probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("router_probabilities must have shape [rows, experts]")
    order = np.argsort(-probabilities, axis=1)
    sorted_probabilities = np.take_along_axis(probabilities, order, axis=1)
    cumulative = np.cumsum(sorted_probabilities, axis=1)
    selected_sorted = cumulative - sorted_probabilities < cumulative_probability
    selected_sorted[:, max_experts:] = False
    selected_sorted[:, 0] = True
    selected = np.zeros_like(selected_sorted, dtype=bool)
    np.put_along_axis(selected, order, selected_sorted, axis=1)
    return selected


def noisy_or_fusion(
    general_probability: np.ndarray,
    expert_probabilities: np.ndarray,
    router_probabilities: np.ndarray,
    selected_mask: np.ndarray,
) -> np.ndarray:
    general_probability = np.asarray(general_probability, dtype=np.float64)
    expert_probabilities = np.asarray(expert_probabilities, dtype=np.float64)
    router_probabilities = np.asarray(router_probabilities, dtype=np.float64)
    selected_mask = np.asarray(selected_mask, dtype=bool)
    if expert_probabilities.shape != router_probabilities.shape:
        raise ValueError("expert and router probability shapes must match")
    routed_evidence = np.where(
        selected_mask,
        router_probabilities * expert_probabilities,
        0.0,
    )
    result = 1.0 - (1.0 - general_probability) * np.prod(
        1.0 - routed_evidence,
        axis=1,
    )
    return np.clip(result, 0.0, 1.0)


def threshold_for_best_f1(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1],
        1e-12,
    )
    return float(thresholds[int(np.nanargmax(f1))])


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 15,
) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probabilities >= edges[index]) & (
                probabilities <= edges[index + 1]
            )
        else:
            selected = (probabilities >= edges[index]) & (
                probabilities < edges[index + 1]
            )
        if not selected.any():
            continue
        accuracy = labels[selected].mean()
        confidence = probabilities[selected].mean()
        ece += selected.mean() * abs(float(accuracy - confidence))
    return float(ece)


def macro_cwe_average_precision(
    target: np.ndarray,
    cwe_index: np.ndarray,
    per_cwe_probabilities: np.ndarray,
    top5_size: int = 5,
) -> tuple[float, dict[str, float]]:
    target = np.asarray(target, dtype=np.int64)
    cwe_index = np.asarray(cwe_index, dtype=np.int64)
    probabilities = np.asarray(per_cwe_probabilities, dtype=np.float64)
    if probabilities.ndim == 1:
        probabilities = np.repeat(probabilities[:, None], top5_size, axis=1)
    values: dict[str, float] = {}
    for index in range(top5_size):
        selected = (target == 0) | ((target == 1) & (cwe_index == index))
        labels = ((target[selected] == 1) & (cwe_index[selected] == index)).astype(
            np.int64
        )
        if labels.sum() == 0:
            values[str(index)] = float("nan")
        else:
            values[str(index)] = float(
                average_precision_score(labels, probabilities[selected, index])
            )
    finite = [value for value in values.values() if math.isfinite(value)]
    return (float(np.mean(finite)) if finite else float("nan"), values)


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities),
        "threshold": float(threshold),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def effective_number_weights(
    counts: Sequence[int],
    beta: float = 0.9999,
) -> np.ndarray:
    counts_array = np.asarray(counts, dtype=np.float64)
    weights = np.zeros_like(counts_array)
    positive = counts_array > 0
    weights[positive] = (1.0 - beta) / (
        1.0 - np.power(beta, counts_array[positive])
    )
    if positive.any():
        weights[positive] *= positive.sum() / weights[positive].sum()
    return weights


def batched(iterable: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(iterable), size):
        yield iterable[start : start + size]
