from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import SOURCE_NAMES, write_json


RAW_COLUMNS = ["processed_func", "flaw_line_index"]


def normalize_indices(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    values: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return ",".join(str(value) for value in sorted(set(values)))


def selected_raw_rows(
    csv_path: Path,
    wanted_indices: np.ndarray,
    chunksize: int,
) -> pd.DataFrame:
    """Read only requested row numbers from a potentially multi-GB raw CSV."""
    wanted = set(int(value) for value in wanted_indices)
    pieces: list[pd.DataFrame] = []
    offset = 0
    for chunk in pd.read_csv(
        csv_path,
        usecols=RAW_COLUMNS,
        chunksize=chunksize,
        low_memory=False,
    ):
        positions = [index for index in wanted if offset <= index < offset + len(chunk)]
        if positions:
            local = np.asarray(positions, dtype=np.int64) - offset
            selected = chunk.iloc[local].copy()
            selected["source_index"] = positions
            pieces.append(selected)
        offset += len(chunk)
    if not pieces:
        return pd.DataFrame(columns=[*RAW_COLUMNS, "source_index"])
    result = pd.concat(pieces, ignore_index=True)
    if result["source_index"].duplicated().any():
        raise ValueError(f"Duplicate raw indices in {csv_path}")
    missing = wanted.difference(result["source_index"].astype(int))
    if missing:
        raise ValueError(f"Could not find {len(missing)} requested rows in {csv_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=50_000)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, low_memory=False)
    test = manifest[manifest["split"] == "test"].copy().reset_index(drop=True)
    if test["row_id"].duplicated().any():
        raise ValueError("Test row_id values must be unique")

    raw_pieces: list[pd.DataFrame] = []
    for source_code, source_name in SOURCE_NAMES.items():
        subset = test[test["source_code"] == source_code]
        if subset.empty:
            continue
        raw = selected_raw_rows(
            args.raw_data_dir / f"{source_name}.csv",
            subset["source_index"].to_numpy(dtype=np.int64),
            args.chunksize,
        )
        raw["source_code"] = source_code
        raw_pieces.append(raw)
        print(json.dumps({"event": "raw_source", "source": source_name, "rows": len(raw)}), flush=True)

    raw = pd.concat(raw_pieces, ignore_index=True)
    merged = test.merge(
        raw,
        on=["source_code", "source_index"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if merged["processed_func"].isna().any():
        raise ValueError("Raw source code is missing after merge")
    merged["flaw_line_indices"] = merged["flaw_line_index"].map(normalize_indices)
    merged["raw_nonempty_lines"] = merged["processed_func"].map(
        lambda value: sum(bool(line.strip()) for line in str(value).split("\n"))
    )
    merged["raw_flaw_count"] = merged["flaw_line_indices"].map(
        lambda value: 0 if not value else len(value.split(","))
    )
    keep = [
        "row_id",
        "source",
        "source_code",
        "source_index",
        "target",
        "cwe",
        "cwe_index",
        "stratum",
        "cve_id",
        "commit_id",
        "project",
        "cluster_id",
        "flaw_line_indices",
        "raw_nonempty_lines",
        "raw_flaw_count",
    ]
    output = merged[keep]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, compression="gzip")

    positive_top5 = output[(output["target"] == 1) & output["cwe_index"].between(0, 4)]
    audit = {
        "manifest": str(args.manifest),
        "test_rows": int(len(output)),
        "test_vulnerable": int(output["target"].sum()),
        "top5_vulnerable_functions": int(len(positive_top5)),
        "top5_with_line_labels": int((positive_top5["raw_flaw_count"] > 0).sum()),
        "raw_nonempty_lines": int(output["raw_nonempty_lines"].sum()),
        "raw_flaw_lines": int(output["raw_flaw_count"].sum()),
        "top5_counts": {
            str(int(index)): int(count)
            for index, count in positive_top5["cwe_index"].value_counts().sort_index().items()
        },
    }
    write_json(args.output.with_suffix(".audit.json"), audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
