from __future__ import annotations

import argparse
import gzip
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import SOURCE_CODES, normalize_cwe, stable_hash, write_json


SOURCE_FILENAMES = {
    "train": "train.csv",
    "valid": "valid.csv",
    "test": "test.csv",
}
INPUT_COLUMNS = {
    "processed_func",
    "target",
    "CWE ID",
    "CVE ID",
    "commit_id",
    "project",
    "Publish Date",
}


def clean_identifier(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def scan_sources(
    data_dir: Path,
    output_path: Path,
    chunksize: int,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    header = True
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as output:
        for source_name, filename in SOURCE_FILENAMES.items():
            source_path = data_dir / filename
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            source_index = 0
            for chunk in pd.read_csv(
                source_path,
                usecols=lambda name: name in INPUT_COLUMNS,
                chunksize=chunksize,
                low_memory=False,
            ):
                missing = INPUT_COLUMNS - set(chunk.columns)
                if missing:
                    raise ValueError(
                        f"{source_path} is missing required columns: {sorted(missing)}"
                    )
                functions = chunk["processed_func"].fillna("").astype(str)
                metadata = pd.DataFrame(
                    {
                        "global_index": np.arange(
                            total_rows,
                            total_rows + len(chunk),
                            dtype=np.int64,
                        ),
                        "source": source_name,
                        "source_code": SOURCE_CODES[source_name],
                        "source_index": np.arange(
                            source_index,
                            source_index + len(chunk),
                            dtype=np.int64,
                        ),
                        "row_id": [
                            f"{source_name}:{index}"
                            for index in range(source_index, source_index + len(chunk))
                        ],
                        "target": chunk["target"].fillna(0).astype(np.int8),
                        "cwe": chunk["CWE ID"].map(normalize_cwe),
                        "cve_id": chunk["CVE ID"].map(clean_identifier),
                        "commit_id": chunk["commit_id"].map(clean_identifier),
                        "project": chunk["project"].map(clean_identifier),
                        "publish_date": chunk["Publish Date"].map(clean_identifier),
                        "func_hash": functions.map(stable_hash),
                    }
                )
                metadata["cluster_id"] = [
                    f"cve:{cve}"
                    if cve
                    else (
                        f"commit:{commit}"
                        if commit
                        else f"function:{function_hash}"
                    )
                    for cve, commit, function_hash in zip(
                        metadata["cve_id"],
                        metadata["commit_id"],
                        metadata["func_hash"],
                    )
                ]
                metadata.to_csv(output, index=False, header=header)
                header = False
                source_index += len(chunk)
                total_rows += len(chunk)
                print(
                    json.dumps(
                        {
                            "event": "scan_progress",
                            "source": source_name,
                            "source_rows": source_index,
                            "total_rows": total_rows,
                        }
                    ),
                    flush=True,
                )
    return pd.read_csv(output_path, low_memory=False)


def choose_top5(metadata: pd.DataFrame, split_column: str) -> list[str]:
    training = metadata[
        (metadata[split_column] == "train")
        & (metadata["target"] == 1)
        & (metadata["cwe"] != "unknown")
    ]
    counts = training["cwe"].value_counts()
    if len(counts) < 5:
        raise ValueError(f"Only {len(counts)} known CWEs are present in training")
    return [str(value) for value in counts.index[:5]]


def add_cwe_labels(
    metadata: pd.DataFrame,
    top5: list[str],
) -> pd.DataFrame:
    result = metadata.copy()
    mapping = {cwe: index for index, cwe in enumerate(top5)}
    result["cwe_index"] = np.int8(-1)
    vulnerable_top5 = (result["target"] == 1) & result["cwe"].isin(top5)
    result.loc[vulnerable_top5, "cwe_index"] = (
        result.loc[vulnerable_top5, "cwe"].map(mapping).astype(np.int8)
    )
    result.loc[
        (result["target"] == 1) & (result["cwe"] == "unknown"),
        "cwe_index",
    ] = -2
    result["stratum"] = "clean"
    vulnerable = result["target"] == 1
    result.loc[vulnerable & result["cwe"].isin(top5), "stratum"] = result.loc[
        vulnerable & result["cwe"].isin(top5), "cwe"
    ]
    result.loc[vulnerable & ~result["cwe"].isin(top5), "stratum"] = "tail"
    result.loc[vulnerable & (result["cwe"] == "unknown"), "stratum"] = "unknown"
    return result


def assign_projects(
    metadata: pd.DataFrame,
    seed: int,
) -> dict[str, str]:
    project_frame = metadata.copy()
    if "project_group" not in project_frame:
        project_frame["project_group"] = project_frame["project"].where(
            project_frame["project"].astype(bool),
            "missing-project:" + project_frame["cluster_id"],
        )
    groups = (
        project_frame.groupby("project_group", sort=False)
        .agg(rows=("target", "size"), positives=("target", "sum"))
    )
    vulnerable = project_frame[
        (project_frame["target"] == 1)
        & (project_frame["cwe"] != "unknown")
    ]
    cwe_counts = pd.crosstab(vulnerable["project_group"], vulnerable["cwe"])
    cwe_totals = cwe_counts.sum(axis=0)
    cwe_group_counts = (cwe_counts > 0).sum(axis=0)
    eligible_cwes = sorted(
        str(cwe)
        for cwe in cwe_counts.columns[
            (cwe_totals >= 30) & (cwe_group_counts >= 3)
        ]
    )
    if eligible_cwes:
        groups = groups.join(
            cwe_counts[eligible_cwes].add_prefix("cwe_count:"),
            how="left",
        ).fillna(0)
    feature_columns = ["rows", "positives"] + [
        f"cwe_count:{cwe}" for cwe in eligible_cwes
    ]
    feature_values = groups[feature_columns].to_numpy(dtype=np.float64)
    totals = feature_values.sum(axis=0)
    split_fractions = {
        "train": 0.8,
        "valid": 0.1,
        "test": 0.1,
    }
    targets = {
        name: fraction * totals for name, fraction in split_fractions.items()
    }
    feature_weights = np.ones(len(feature_columns), dtype=np.float64)
    if len(feature_columns) > 2:
        feature_weights[2:] = 0.5
    current = {
        name: np.zeros(len(feature_columns), dtype=np.float64)
        for name in targets
    }
    assignments: dict[str, str] = {}
    rng = np.random.default_rng(seed)
    groups = groups.reset_index()
    groups["jitter"] = rng.uniform(0.0, 1e-6, size=len(groups))
    groups["priority"] = np.max(
        groups[feature_columns].to_numpy(dtype=np.float64)
        / np.maximum(targets["train"], 1.0),
        axis=1,
    )
    groups = groups.sort_values(
        ["priority", "rows", "positives", "jitter"],
        ascending=[False, False, False, False],
    )

    def objective(
        candidate_split: str,
        addition: np.ndarray,
    ) -> float:
        trial = {name: values.copy() for name, values in current.items()}
        trial[candidate_split] += addition
        score = 0.0
        for name, target in targets.items():
            relative = (trial[name] - target) / np.maximum(target, 1.0)
            score += float((feature_weights * np.square(relative)).sum())
            overfill = np.maximum(trial[name] - target, 0.0) / np.maximum(
                target,
                1.0,
            )
            score += 4.0 * float(
                (feature_weights * np.square(overfill)).sum()
            )
        return score

    for row in groups.itertuples(index=False):
        row_values = groups.loc[
            groups["project_group"] == row.project_group,
            feature_columns,
        ].iloc[0]
        addition = row_values.to_numpy(dtype=np.float64)
        scores = {
            split: objective(split, addition)
            for split in ("train", "valid", "test")
        }
        selected = min(scores, key=lambda name: (scores[name], name))
        assignments[str(row.project_group)] = selected
        current[selected] += addition

    if set(assignments.values()) != {"train", "valid", "test"}:
        raise RuntimeError("Project assignment produced an empty split")
    return assignments


def connected_project_groups(metadata: pd.DataFrame) -> pd.Series:
    base_groups = metadata["project"].where(
        metadata["project"].astype(bool),
        "missing-project:" + metadata["cluster_id"],
    )
    parent = {group: group for group in base_groups.unique()}

    def find(group: str) -> str:
        root = group
        while parent[root] != root:
            root = parent[root]
        while parent[group] != group:
            next_group = parent[group]
            parent[group] = root
            group = next_group
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    working = metadata.assign(_base_group=base_groups)
    for key in ("cve_id", "commit_id", "func_hash"):
        pairs = working.loc[
            working[key].astype(bool),
            ["_base_group", key],
        ].drop_duplicates()
        first_group: dict[str, str] = {}
        for group, identifier in pairs.itertuples(index=False, name=None):
            group = str(group)
            identifier = str(identifier)
            if identifier in first_group:
                union(group, first_group[identifier])
            else:
                first_group[identifier] = group
    roots = {group: find(group) for group in parent}
    return base_groups.map(roots)


def pairwise_overlap(
    metadata: pd.DataFrame,
    split_column: str,
    key: str,
) -> dict[str, int]:
    values = {
        split: set(
            metadata.loc[
                (metadata[split_column] == split) & metadata[key].astype(bool),
                key,
            ].astype(str)
        )
        for split in ("train", "valid", "test")
    }
    return {
        f"{left}_{right}": len(values[left] & values[right])
        for left, right in itertools.combinations(values, 2)
    }


def describe_protocol(
    metadata: pd.DataFrame,
    split_column: str,
    top5: list[str],
) -> dict[str, object]:
    split_summary: dict[str, object] = {}
    for split in ("train", "valid", "test"):
        selected = metadata[metadata[split_column] == split]
        positives = selected[selected["target"] == 1]
        split_summary[split] = {
            "rows": int(len(selected)),
            "positives": int(positives.shape[0]),
            "positive_rate": float(positives.shape[0] / max(len(selected), 1)),
            "unknown_cwe_positives": int((positives["cwe"] == "unknown").sum()),
            "top5_positive_counts": {
                cwe: int((positives["cwe"] == cwe).sum()) for cwe in top5
            },
            "projects": int(selected["project"].nunique()),
            "clusters": int(selected["cluster_id"].nunique()),
        }
    return {
        "top5": top5,
        "splits": split_summary,
        "cross_split_overlap": {
            key: pairwise_overlap(metadata, split_column, key)
            for key in ("project", "cve_id", "commit_id", "func_hash")
        },
        "duplicate_function_rows": int(metadata["func_hash"].duplicated().sum()),
    }


def write_protocol(
    metadata: pd.DataFrame,
    split_column: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    top5 = choose_top5(metadata, split_column)
    labeled = add_cwe_labels(metadata, top5)
    labeled["split"] = labeled[split_column]
    columns = [
        "global_index",
        "source",
        "source_code",
        "source_index",
        "row_id",
        "split",
        "target",
        "cwe",
        "cwe_index",
        "stratum",
        "cve_id",
        "commit_id",
        "project",
        "publish_date",
        "func_hash",
        "cluster_id",
    ]
    labeled[columns].to_csv(
        output_dir / "manifest.csv.gz",
        index=False,
        compression="gzip",
    )
    write_json(output_dir / "top5.json", top5)
    write_json(
        output_dir / "audit.json",
        describe_protocol(labeled, "split", top5),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=2_000)
    parser.add_argument("--project-seed", type=int, default=20260729)
    parser.add_argument("--reuse-scan", action="store_true")
    args = parser.parse_args()

    raw_metadata_path = args.output_dir / "raw_metadata.csv.gz"
    if args.reuse_scan and raw_metadata_path.is_file():
        metadata = pd.read_csv(raw_metadata_path, low_memory=False)
    else:
        metadata = scan_sources(
            args.data_dir,
            raw_metadata_path,
            chunksize=args.chunksize,
        )

    metadata["official_split"] = metadata["source"]
    write_protocol(
        metadata,
        split_column="official_split",
        output_dir=args.output_dir / "official",
    )

    project_frame = metadata.copy()
    project_frame["project_group"] = connected_project_groups(project_frame)
    assignments = assign_projects(project_frame, seed=args.project_seed)
    project_frame["project_split"] = project_frame["project_group"].map(assignments)
    if project_frame["project_split"].isna().any():
        raise RuntimeError("Some rows did not receive a project split")
    write_protocol(
        project_frame,
        split_column="project_split",
        output_dir=args.output_dir / "project",
    )

    write_json(
        args.output_dir / "preparation.json",
        {
            "rows": int(len(metadata)),
            "source_rows": {
                source: int((metadata["source"] == source).sum())
                for source in SOURCE_FILENAMES
            },
            "project_seed": args.project_seed,
            "data_dir": str(args.data_dir),
        },
    )


if __name__ == "__main__":
    main()
