from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_line_specialization import (
    PROTOCOLS,
    SEEDS,
    cluster_bootstrap,
    function_metrics,
    prefix_columns,
    summarize,
)
from common import read_json, write_json


def oracle_pair(
    g1_line_root: Path,
    onehot_line_root: Path,
    protocol: str,
    seed: int,
) -> pd.DataFrame:
    g1 = function_metrics(
        g1_line_root / protocol / f"seed{seed}" / "g1" / "line_scores.csv.gz"
    )
    experts: list[pd.DataFrame] = []
    for index in range(5):
        frame = function_metrics(
            onehot_line_root
            / protocol
            / f"seed{seed}"
            / f"expert{index}"
            / "line_scores.csv.gz"
        )
        experts.append(frame[frame["cwe_index"] == index].copy())
    expert = pd.concat(experts, ignore_index=True)
    # A conditional CWE head is not a vulnerability detector. Suppress its
    # function prediction so only G1 is used for end-to-end vulnerability gates.
    expert["function_probability"] = np.nan
    expert["function_prediction"] = -1
    pair = prefix_columns(g1, "g1").merge(
        prefix_columns(expert, "onehot_oracle"),
        on=["row_id", "cluster_id", "cwe", "cwe_index", "target"],
        how="inner",
        validate="one_to_one",
    )
    return pair[
        (pair["target"] == 1) & pair["cwe_index"].between(0, 4)
    ].copy()


def routed_onehot(
    onehot_line_root: Path,
    runs_root: Path,
    protocol: str,
    seed: int,
) -> pd.DataFrame:
    expert_frames: list[pd.DataFrame] = []
    for index in range(5):
        frame = function_metrics(
            onehot_line_root
            / protocol
            / f"seed{seed}"
            / f"expert{index}"
            / "line_scores.csv.gz"
        )
        frame["selected_expert"] = index
        expert_frames.append(frame)
    experts = pd.concat(expert_frames, ignore_index=True)
    predictions = pd.read_csv(
        runs_root / protocol / f"onehot-seed{seed}" / "predictions.csv.gz",
        low_memory=False,
    )
    selected = predictions[
        [
            "row_id",
            "selected_expert",
            "selected_experts_top3",
            "probability",
            "prediction",
        ]
    ]
    result = selected.merge(
        experts,
        on=["row_id", "selected_expert"],
        how="left",
        validate="one_to_one",
    )
    result["function_probability"] = result["probability"]
    result["function_prediction"] = result["prediction"]
    return result


def prefixed_line_metrics(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    columns = ("line_ap", "mrr", "ifa", "top1", "top3", "top5", "top10")
    renamed = frame.rename(columns={column: f"{prefix}_{column}" for column in columns})
    renamed[f"{prefix}_function_probability"] = renamed["function_probability"]
    renamed[f"{prefix}_function_prediction"] = renamed["function_prediction"]
    return renamed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--g1-line-root", type=Path, required=True)
    parser.add_argument("--onehot-line-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--protocol", action="append", choices=PROTOCOLS)
    parser.add_argument("--seed", action="append", type=int)
    args = parser.parse_args()

    protocols = tuple(args.protocol) if args.protocol else PROTOCOLS
    seeds = tuple(args.seed) if args.seed else SEEDS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    table_rows: list[dict[str, object]] = []

    for protocol in protocols:
        top5 = read_json(args.prepared_root / protocol / "top5.json")
        for seed in seeds:
            pair = oracle_pair(
                args.g1_line_root,
                args.onehot_line_root,
                protocol,
                seed,
            )
            eligible = (pair["g1_verified_flaws"] > 0) & (
                pair["onehot_oracle_verified_flaws"] > 0
            )
            all_gt = pair[eligible].copy()
            g1_tp = all_gt[all_gt["g1_function_prediction"] == 1].copy()
            bootstrap = cluster_bootstrap(
                all_gt,
                "g1_line_ap",
                "onehot_oracle_line_ap",
                args.bootstrap,
                seed,
            )

            routed = routed_onehot(
                args.onehot_line_root,
                args.runs_root,
                protocol,
                seed,
            )
            routed = routed[
                (routed["target"] == 1)
                & routed["cwe_index"].between(0, 4)
                & (routed["verified_flaws"] > 0)
            ].copy()
            routed["route_correct"] = (
                routed["selected_expert"] == routed["cwe_index"]
            )
            routed["triple_top10"] = (
                (routed["function_prediction"] == 1)
                & routed["route_correct"]
                & (routed["top10"] == 1)
            ).astype(np.int8)
            routed_summary = summarize(
                prefixed_line_metrics(routed, "onehot"),
                "onehot",
            )
            correct_route_summary = summarize(
                prefixed_line_metrics(routed[routed["route_correct"]], "onehot"),
                "onehot",
            )
            ensemble_run = read_json(
                args.runs_root / protocol / f"onehot-seed{seed}" / "run.json"
            )
            routing = ensemble_run["test_metrics"]["routing"]

            result = {
                "protocol": protocol,
                "seed": seed,
                "top5": top5,
                "all_gt": {
                    "g1": summarize(all_gt, "g1"),
                    "onehot_oracle": summarize(all_gt, "onehot_oracle"),
                    "paired_bootstrap_line_ap": bootstrap,
                },
                "g1_tp": {
                    "g1": summarize(g1_tp, "g1"),
                    "onehot_oracle": summarize(g1_tp, "onehot_oracle"),
                },
                "routing": routing,
                "onehot_routed_all_gt": routed_summary,
                "onehot_correct_route_all_gt": correct_route_summary,
                "triple_vulnerability_cwe_line_top10": float(
                    routed["triple_top10"].mean()
                ),
            }
            write_json(
                args.output_dir / f"summary-{protocol}-seed{seed}.json",
                result,
            )
            routed.to_csv(
                args.output_dir / f"routed-functions-{protocol}-seed{seed}.csv.gz",
                index=False,
                compression="gzip",
            )
            results.append(result)
            table_rows.append(
                {
                    "protocol": protocol,
                    "seed": seed,
                    "functions": len(all_gt),
                    "g1_macro_line_ap": result["all_gt"]["g1"]["macro_line_ap"],
                    "onehot_oracle_macro_line_ap": result["all_gt"]["onehot_oracle"]["macro_line_ap"],
                    "oracle_difference": bootstrap["difference"],
                    "ci95_low": bootstrap["ci95_low"],
                    "ci95_high": bootstrap["ci95_high"],
                    "route_top1": routing["top1_accuracy"],
                    "route_top3": routing["top3_accuracy"],
                    "routed_macro_line_ap": routed_summary["macro_line_ap"],
                    "triple_top10": result["triple_vulnerability_cwe_line_top10"],
                }
            )

    table = pd.DataFrame(table_rows)
    table.to_csv(args.output_dir / "primary-summary.csv", index=False)
    write_json(args.output_dir / "all-results.json", {"runs": results})
    averages = table.groupby("protocol", as_index=False).mean(numeric_only=True)
    averages.to_csv(args.output_dir / "protocol-means.csv", index=False)

    report = [
        "# 条件付きone-hot CWE専門家の行レベル検知",
        "",
        "G1が脆弱性を二値判定し、one-hot専門家5個のargmaxがTop5 CWEを選び、選ばれた専門家のattentionで行を順位付けする。",
        "専門家はTop5脆弱関数だけで、対象CWEを陽性、他のTop5 CWEを陰性として学習した。clean・tail・unknownはCWE分類の学習対象外である。",
        "",
        "| 分割 | seed | G1 line AP | oracle one-hot | 差 [95% CI] | CWE top-1 | top-3 | routed line AP | 三段階Top10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        report.append(
            f"| {row.protocol} | {row.seed} | {row.g1_macro_line_ap:.4f} | "
            f"{row.onehot_oracle_macro_line_ap:.4f} | {row.oracle_difference:+.4f} "
            f"[{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] | "
            f"{row.route_top1:.4f} | {row.route_top3:.4f} | "
            f"{row.routed_macro_line_ap:.4f} | {row.triple_top10:.4f} |"
        )
    report.extend(
        [
            "",
            "`oracle one-hot`は正解CWEを既知として専門家を選ぶ上限評価、`routed`はone-hot予測で専門家を選ぶ実運用評価である。",
            "三段階Top10は、G1が脆弱と判定し、CWE top-1が正しく、正解行が上位10行に入った割合である。",
            "",
        ]
    )
    (args.output_dir / "REPORT.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    print(json.dumps({"runs": len(results), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
