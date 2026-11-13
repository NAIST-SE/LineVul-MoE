from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from common import SOURCE_NAMES, sigmoid, write_json
from models import build_model


def parse_flaw_indices(value: object) -> set[int]:
    """Parse BigVul's comma-separated, zero-based physical line indices."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    result: set[int] = set()
    for part in str(value).split(","):
        part = part.strip()
        if part:
            result.add(int(part))
    return result


def token_lines(
    tokens: list[str],
    attention: np.ndarray,
    flaw_indices: set[int],
) -> list[tuple[int, float, int]]:
    """Aggregate token attention into non-empty physical source lines.

    ``flaw_line_index`` counts blank physical lines. RoBERTa can encode several
    consecutive newlines as one token (for example ``ĊĊ``), so each marker
    advances the physical line index even though blank LOC are not ranked.
    The attention assigned to a newline token is included in the preceding
    line, matching LineVul's original token-to-line aggregation.
    """
    rows: list[tuple[int, float, int]] = []
    score_sum = 0.0
    line_index = 0
    has_content = False

    def finish_line() -> None:
        nonlocal score_sum, has_content
        if has_content:
            rows.append(
                (line_index, float(score_sum), int(line_index in flaw_indices))
            )
        score_sum = 0.0
        has_content = False

    for token, score in zip(tokens, attention):
        if token in {"</s>", "<pad>"}:
            break
        if token == "<s>":
            continue
        newline_count = token.count("Ċ")
        is_only_newlines = newline_count > 0 and not token.strip().replace("Ċ", "")
        if is_only_newlines:
            score_sum += float(score)
            for _ in range(newline_count):
                finish_line()
                line_index += 1
            continue
        score_sum += float(score)
        if token.replace("Ġ", "").strip():
            has_content = True
    finish_line()
    return rows


def cache_batch(frame: pd.DataFrame, caches: dict[int, np.ndarray]) -> np.ndarray:
    """Fetch a batch that may contain rows from train, valid, and test caches."""
    values = [
        np.asarray(
            caches[int(row.source_code)][int(row.source_index)],
            dtype=np.int64,
        )
        for row in frame.itertuples(index=False)
    ]
    return np.stack(values, axis=0)


def ranked_metrics(
    lines: pd.DataFrame,
    total_lines: int,
    total_flaws: int,
    effort_recall: float = 0.2,
    recall_loc: float = 0.01,
) -> dict[str, float | int]:
    """Keep the original LineVul end-to-end budget metrics for reference."""
    positive = lines[lines["function_prediction"] == 1].sort_values(
        "score", ascending=False, kind="stable"
    )
    negative = lines[lines["function_prediction"] == 0].sort_values(
        "score", ascending=True, kind="stable"
    )

    target_flaws = int(total_flaws * effort_recall)
    caught = 0
    inspected = 0
    for label in positive["label"].to_numpy(dtype=np.int8):
        inspected += 1
        caught += int(label)
        if caught >= target_flaws:
            break
    effort = inspected / total_lines if total_lines else float("nan")

    budget = int(total_lines * recall_loc)
    caught = 0
    inspected_for_recall = 0
    for ranked in (positive, negative):
        for label in ranked["label"].to_numpy(dtype=np.int8):
            if inspected_for_recall >= budget:
                break
            inspected_for_recall += 1
            caught += int(label)
        if inspected_for_recall >= budget:
            break
    recall = caught / total_flaws if total_flaws else float("nan")
    return {
        "effort_at_20_percent_recall": float(effort),
        "inspected_lines_for_20_percent_recall": int(inspected),
        "recall_at_1_percent_loc": float(recall),
        "line_budget_at_1_percent_loc": int(budget),
        "ranked_positive_lines": int(len(positive)),
        "ranked_negative_lines": int(len(negative)),
    }


def local_metrics(lines: pd.DataFrame) -> dict[str, float | int]:
    if lines.empty:
        return {"functions_with_verified_flaws": 0}
    ifas: list[int] = []
    top10: list[int] = []
    for _, function_lines in lines.groupby("row_id", sort=False):
        labels = function_lines["label"].to_numpy(dtype=np.int8)
        if labels.sum() == 0:
            continue
        ranked = function_lines.sort_values(
            ["score", "line_index"], ascending=[False, True], kind="stable"
        )["label"].to_numpy(dtype=np.int8)
        first = int(np.flatnonzero(ranked)[0])
        ifas.append(first)
        top10.append(int(first < 10))
    return {
        "functions_with_verified_flaws": len(ifas),
        "mean_ifa": float(np.mean(ifas)) if ifas else float("nan"),
        "median_ifa": float(np.median(ifas)) if ifas else float("nan"),
        "top10_function_accuracy": float(np.mean(top10)) if top10 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--line-data", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=("official", "project"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    line_data = pd.read_csv(args.line_data, low_memory=False)
    if args.max_rows is not None:
        line_data = line_data.iloc[:args.max_rows].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    model = build_model("generalist", args.model_path, adapter_size=64)
    if args.weights.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(args.weights))
    else:
        state = torch.load(args.weights, map_location="cpu")["model_state"]
    model.load_state_dict(state, strict=True)
    del state
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    caches = {
        code: np.load(args.cache_dir / f"{name}.input_ids.npy", mmap_mode="r")
        for code, name in SOURCE_NAMES.items()
    }

    output_rows: list[dict[str, Any]] = []
    function_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(line_data), args.batch_size):
            stop = min(start + args.batch_size, len(line_data))
            batch = line_data.iloc[start:stop]
            input_ids = torch.as_tensor(cache_batch(batch, caches), device=device)
            attention_mask = input_ids.ne(tokenizer.pad_token_id)
            encoded = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True,
            )
            first_layer = encoded.attentions[0]
            token_attention = first_layer.sum(dim=1).sum(dim=1).cpu().numpy()
            ids = input_ids.cpu().numpy()

            logits = model.classifier(encoded.last_hidden_state).flatten().cpu().numpy()
            probabilities = sigmoid(logits / args.temperature)
            for offset, meta in enumerate(batch.itertuples(index=False)):
                probability = float(probabilities[offset])
                prediction = int(probability >= args.threshold)
                function_rows.append(dict(row_id=meta.row_id, target=int(meta.target),
                                          cwe_index=int(meta.cwe_index), cve_id=meta.cve_id,
                                          probability=probability, prediction=prediction))
                tokens = tokenizer.convert_ids_to_tokens(ids[offset].tolist())
                values = token_attention[offset].astype(np.float64, copy=True)
                values[0] = 0.0
                last_real = int(attention_mask[offset].sum().item()) - 1
                if last_real >= 0:
                    values[last_real] = 0.0
                flaws = parse_flaw_indices(meta.flaw_line_indices)
                for line_index, score, label in token_lines(tokens, values, flaws):
                    output_rows.append(
                        {
                            "row_id": meta.row_id,
                            "cluster_id": meta.cluster_id,
                            "cwe": meta.cwe,
                            "cwe_index": int(meta.cwe_index),
                            "source_code": int(meta.source_code),
                            "source_index": int(meta.source_index),
                            "line_index": line_index,
                            "score": score,
                            "label": label,
                            "function_target": int(meta.target),
                            "function_probability": probability,
                            "function_prediction": prediction,
                            "raw_nonempty_lines": int(meta.raw_nonempty_lines),
                            "raw_flaw_count": int(meta.raw_flaw_count),
                        }
                    )
            if stop == len(line_data) or stop % 1000 < args.batch_size:
                print(json.dumps({"event": "line_progress", "rows": stop, "total": len(line_data)}), flush=True)

    lines = pd.DataFrame(output_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lines.to_csv(args.output_dir / "line_scores.csv.gz", index=False, compression="gzip")
    pd.DataFrame(function_rows).to_csv(args.output_dir / "predictions.csv.gz", index=False, compression="gzip")
    total_lines = int(line_data["raw_nonempty_lines"].sum())
    total_flaws = int(line_data["raw_flaw_count"].sum())
    result = {
        "kind": "linevul_first_layer_attention",
        "weights": str(args.weights),
        "seed": args.seed,
        "protocol": args.protocol,
        "evaluated_functions": int(len(line_data)),
        "raw_total_nonempty_lines": total_lines,
        "raw_total_flaw_lines": total_flaws,
        "tokenized_ranked_lines": int(len(lines)),
        "verified_flaw_lines_in_window": int(lines["label"].sum()) if not lines.empty else 0,
        "global_metrics_reference_only": ranked_metrics(lines, total_lines, total_flaws),
        "local_metrics_reference_only": local_metrics(lines),
        "elapsed_seconds": time.monotonic() - started,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "max_rows": args.max_rows,
    }
    write_json(args.output_dir / "run.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
