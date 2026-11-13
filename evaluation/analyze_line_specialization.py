from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import read_json, write_json


PROTOCOLS = ("official", "project")
SEEDS = (123456, 42, 2026, 31415, 27182)
TOP_K = (1, 3, 5, 10)


def function_metrics(path: Path) -> pd.DataFrame:
    lines = pd.read_csv(path, low_memory=False)
    # Line-localization metrics are defined only for vulnerable functions with
    # a known Top5 CWE in this study. Filtering before Python-level grouping
    # avoids aggregating tens of thousands of clean/tail functions repeatedly.
    lines = lines[
        (lines["function_target"] == 1) & lines["cwe_index"].between(0, 4)
    ].copy()
    rows: list[dict[str, object]] = []
    for row_id, frame in lines.groupby("row_id", sort=False):
        frame = frame.sort_values(
            ["score", "line_index"], ascending=[False, True], kind="stable"
        )
        labels = frame["label"].to_numpy(dtype=np.int8)
        positives = np.flatnonzero(labels)
        first = int(positives[0]) if positives.size else -1
        base = frame.iloc[0]
        record: dict[str, object] = {
            "row_id": row_id,
            "cluster_id": base["cluster_id"],
            "cwe": base["cwe"],
            "cwe_index": int(base["cwe_index"]),
            "target": int(base["function_target"]),
            "function_probability": float(base["function_probability"]),
            "function_prediction": int(base["function_prediction"]),
            "ranked_lines": int(len(frame)),
            "verified_flaws": int(labels.sum()),
            "line_ap": (
                float(average_precision_score(labels, frame["score"].to_numpy()))
                if positives.size
                else float("nan")
            ),
            "ifa": float(first) if positives.size else float("nan"),
            "mrr": float(1.0 / (first + 1)) if positives.size else float("nan"),
        }
        for k in TOP_K:
            record[f"top{k}"] = int(first >= 0 and first < k)
        rows.append(record)
    return pd.DataFrame(rows)


def macro_metric(frame: pd.DataFrame, column: str) -> tuple[float, dict[str, float]]:
    per_cwe: dict[str, float] = {}
    for index in range(5):
        values = frame.loc[frame["cwe_index"] == index, column]
        per_cwe[str(index)] = float(values.mean()) if len(values) else float("nan")
    finite = [value for value in per_cwe.values() if math.isfinite(value)]
    return (float(np.mean(finite)) if finite else float("nan"), per_cwe)


def summarize(frame: pd.DataFrame, prefix: str) -> dict[str, object]:
    frame = frame.copy()
    result: dict[str, object] = {"functions": int(len(frame))}
    metrics = ["line_ap", "mrr", "ifa", "top1", "top3", "top5", "top10"]
    prediction_column = (
        f"{prefix}_function_prediction"
        if f"{prefix}_function_prediction" in frame
        else "function_prediction"
    )
    if prediction_column in frame and (frame[prediction_column] >= 0).all():
        for k in TOP_K:
            metric = f"e2e_top{k}"
            frame[f"{prefix}_{metric}"] = (
                (frame[prediction_column] == 1) & (frame[f"{prefix}_top{k}"] == 1)
            ).astype(np.int8)
            metrics.append(metric)
    for metric in metrics:
        macro, per_cwe = macro_metric(frame, f"{prefix}_{metric}")
        result[f"macro_{metric}"] = macro
        result[f"per_cwe_{metric}"] = per_cwe
    return result


def cluster_bootstrap(
    frame: pd.DataFrame,
    left: str,
    right: str,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    """Paired bootstrap of macro-CWE means, resampling CVE/commit clusters."""
    cluster_codes, clusters = pd.factorize(frame["cluster_id"], sort=True)
    n_clusters = len(clusters)
    left_sum = np.zeros((n_clusters, 5), dtype=np.float64)
    right_sum = np.zeros((n_clusters, 5), dtype=np.float64)
    counts = np.zeros((n_clusters, 5), dtype=np.float64)
    cwe = frame["cwe_index"].to_numpy(dtype=np.int64)
    np.add.at(left_sum, (cluster_codes, cwe), frame[left].to_numpy(dtype=np.float64))
    np.add.at(right_sum, (cluster_codes, cwe), frame[right].to_numpy(dtype=np.float64))
    np.add.at(counts, (cluster_codes, cwe), 1.0)
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=np.float64)
    probability = np.full(n_clusters, 1.0 / n_clusters)
    for index in range(iterations):
        weights = rng.multinomial(n_clusters, probability).astype(np.float64)
        denominator = weights @ counts
        left_means = np.divide(
            weights @ left_sum,
            denominator,
            out=np.full(5, np.nan),
            where=denominator > 0,
        )
        right_means = np.divide(
            weights @ right_sum,
            denominator,
            out=np.full(5, np.nan),
            where=denominator > 0,
        )
        differences[index] = np.nanmean(right_means - left_means)
    observed_left, _ = macro_metric(frame, left)
    observed_right, _ = macro_metric(frame, right)
    return {
        "clusters": n_clusters,
        "iterations": iterations,
        "difference": observed_right - observed_left,
        "ci95_low": float(np.nanquantile(differences, 0.025)),
        "ci95_high": float(np.nanquantile(differences, 0.975)),
    }


def prefix_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    identity = {"row_id", "cluster_id", "cwe", "cwe_index", "target"}
    return frame.rename(
        columns={column: f"{prefix}_{column}" for column in frame.columns if column not in identity}
    )


def oracle_pair(line_root: Path, protocol: str, seed: int) -> pd.DataFrame:
    seed_root = line_root / protocol / f"seed{seed}"
    g1 = prefix_columns(function_metrics(seed_root / "g1" / "line_scores.csv.gz"), "g1")
    experts: list[pd.DataFrame] = []
    for index in range(5):
        frame = function_metrics(seed_root / f"expert{index}" / "line_scores.csv.gz")
        frame = frame[frame["cwe_index"] == index].copy()
        experts.append(prefix_columns(frame, "expert"))
    expert = pd.concat(experts, ignore_index=True)
    pair = g1.merge(
        expert,
        on=["row_id", "cluster_id", "cwe", "cwe_index", "target"],
        how="inner",
        validate="one_to_one",
    )
    return pair[(pair["target"] == 1) & pair["cwe_index"].between(0, 4)].copy()


def population_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    evaluable = (frame["g1_verified_flaws"] > 0) & (frame["expert_verified_flaws"] > 0)
    return {
        "all_gt": evaluable,
        "common_tp": evaluable & (frame["g1_function_prediction"] == 1) & (frame["expert_function_prediction"] == 1),
        "g1_tp": evaluable & (frame["g1_function_prediction"] == 1),
        "expert_tp": evaluable & (frame["expert_function_prediction"] == 1),
    }


def routed_e5(
    line_root: Path,
    runs_root: Path,
    protocol: str,
    seed: int,
) -> pd.DataFrame:
    expert_frames: list[pd.DataFrame] = []
    for index in range(5):
        frame = function_metrics(
            line_root / protocol / f"seed{seed}" / f"expert{index}" / "line_scores.csv.gz"
        )
        frame["selected_expert"] = index
        expert_frames.append(frame)
    all_experts = pd.concat(expert_frames, ignore_index=True)
    predictions = pd.read_csv(
        runs_root / protocol / f"e5-seed{seed}-max" / "predictions.csv.gz",
        low_memory=False,
    )
    probability_columns = [f"cwe_probability_{index}" for index in range(5)]
    predictions["selected_expert"] = np.argmax(
        predictions[probability_columns].to_numpy(dtype=np.float64), axis=1
    )
    selected = predictions[["row_id", "selected_expert", "probability", "prediction"]]
    result = selected.merge(
        all_experts,
        on=["row_id", "selected_expert"],
        how="left",
        validate="one_to_one",
        suffixes=("_e5", "_expert"),
    )
    result["function_probability"] = result["probability"]
    result["function_prediction"] = result["prediction"]
    return result


def ensemble_g5(line_root: Path, protocol: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        path = line_root / protocol / f"seed{seed}" / "g1" / "line_scores.csv.gz"
        lines = pd.read_csv(path, low_memory=False)
        lines = lines[
            (lines["function_target"] == 1) & lines["cwe_index"].between(0, 4)
        ].copy()
        totals = lines.groupby("row_id")["score"].transform("sum")
        lines[f"normalized_{seed}"] = np.divide(
            lines["score"], totals, out=np.zeros(len(lines)), where=totals.to_numpy() != 0
        )
        keep = ["row_id", "line_index", f"normalized_{seed}"]
        if not frames:
            keep = [
                "row_id", "line_index", "cluster_id", "cwe", "cwe_index", "function_target",
                "label", "function_probability", "function_prediction", f"normalized_{seed}",
            ]
        frames.append(lines[keep])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["row_id", "line_index"], how="inner", validate="one_to_one")
    merged["score"] = merged[[f"normalized_{seed}" for seed in SEEDS]].mean(axis=1)
    # Function-level G5 predictions are added separately by the caller if needed.
    temporary = merged.copy()
    temporary["function_probability"] = np.nan
    temporary["function_prediction"] = -1
    path = line_root / protocol / "g5-line-scores.csv.gz"
    temporary.to_csv(path, index=False, compression="gzip")
    return function_metrics(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--line-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--protocol", action="append", choices=PROTOCOLS)
    parser.add_argument("--seed", action="append", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocols = tuple(args.protocol) if args.protocol else PROTOCOLS
    seeds = tuple(args.seed) if args.seed else SEEDS

    all_results: list[dict[str, object]] = []
    audit: dict[str, object] = {}
    for protocol in protocols:
        top5 = read_json(args.prepared_root / protocol / "top5.json")
        audit[protocol] = {"top5": top5}
        for seed in seeds:
            pair = oracle_pair(args.line_root, protocol, seed)
            pair.to_csv(
                args.output_dir / f"oracle-functions-{protocol}-seed{seed}.csv.gz",
                index=False,
                compression="gzip",
            )
            masks = population_masks(pair)
            seed_result: dict[str, object] = {
                "protocol": protocol,
                "seed": seed,
                "top5": top5,
                "raw_top5_functions": int(len(pair)),
                "populations": {},
            }
            for name, mask in masks.items():
                selected = pair[mask].copy()
                population = {
                    "g1": summarize(selected, "g1"),
                    "oracle_expert": summarize(selected, "expert"),
                }
                if name in {"all_gt", "common_tp"} and len(selected):
                    population["paired_bootstrap_line_ap"] = cluster_bootstrap(
                        selected,
                        "g1_line_ap",
                        "expert_line_ap",
                        args.bootstrap,
                        seed + (0 if name == "all_gt" else 1),
                    )
                seed_result["populations"][name] = population

            routed = routed_e5(args.line_root, args.runs_root, protocol, seed)
            routed = routed[
                (routed["target"] == 1)
                & routed["cwe_index"].between(0, 4)
                & (routed["verified_flaws"] > 0)
            ].copy()
            seed_result["e5_max_routed_all_gt"] = summarize(
                routed.rename(columns={column: f"routed_{column}" for column in (
                    "line_ap", "mrr", "ifa", "top1", "top3", "top5", "top10"
                )}),
                "routed",
            )
            write_json(args.output_dir / f"summary-{protocol}-seed{seed}.json", seed_result)
            all_results.append(seed_result)

        if set(SEEDS).issubset(seeds):
            g5 = ensemble_g5(args.line_root, protocol)
            g5 = g5[(g5["target"] == 1) & g5["cwe_index"].between(0, 4) & (g5["verified_flaws"] > 0)]
            audit[protocol]["g5_line_ensemble"] = summarize(
                g5.rename(columns={column: f"g5_{column}" for column in (
                    "line_ap", "mrr", "ifa", "top1", "top3", "top5", "top10"
                )}),
                "g5",
            )

    write_json(args.output_dir / "all-results.json", {"runs": all_results, "additional": audit})
    rows: list[dict[str, object]] = []
    for result in all_results:
        primary = result["populations"]["all_gt"]
        bootstrap = primary["paired_bootstrap_line_ap"]
        rows.append(
            {
                "protocol": result["protocol"],
                "seed": result["seed"],
                "functions": primary["g1"]["functions"],
                "g1_macro_line_ap": primary["g1"]["macro_line_ap"],
                "oracle_macro_line_ap": primary["oracle_expert"]["macro_line_ap"],
                "difference": bootstrap["difference"],
                "ci95_low": bootstrap["ci95_low"],
                "ci95_high": bootstrap["ci95_high"],
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "primary-summary.csv", index=False)
    report = [
        "# CWE専門家による行レベル検知実験",
        "",
        "主解析は、正解CWEがTop5に属し、正解行が512-token窓内で確認できた全脆弱関数を対象とする。",
        "oracle専門家は正解CWEで専門家を選ぶため、専門化そのものの効果を測る上限評価であり、実運用性能ではない。",
        "",
        "| 分割 | seed | 関数数 | G1 macro line AP | oracle専門家 | 差 | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        report.append(
            f"| {row.protocol} | {row.seed} | {row.functions} | {row.g1_macro_line_ap:.4f} | "
            f"{row.oracle_macro_line_ap:.4f} | {row.difference:+.4f} | "
            f"[{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] |"
        )
    report.extend(["", "TP限定、E5最大スコア経路、G5行アンサンブルの結果は `all-results.json` に保存した。", ""])
    (args.output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"runs": len(all_results), "output": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
