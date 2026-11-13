from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--csv", type=Path)
    inputs.add_argument("--parquet", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", choices=("train", "valid", "test"), required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--csv-chunksize", type=int, default=1_000)
    parser.add_argument("--tokenizer-batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(
        args.manifest,
        usecols=["source", "source_index"],
        low_memory=False,
    )
    source_rows = manifest[manifest["source"] == args.source]
    row_count = int(len(source_rows))
    expected_indices = np.arange(row_count, dtype=np.int64)
    observed_indices = np.sort(source_rows["source_index"].to_numpy(dtype=np.int64))
    if not np.array_equal(observed_indices, expected_indices):
        raise ValueError(f"Manifest indices for {args.source} are not contiguous")

    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.is_file() and metadata_path.is_file() and not args.force:
        existing = np.load(args.output, mmap_mode="r")
        if existing.shape == (row_count, args.block_size):
            print(f"verified existing cache {args.output} {existing.shape}")
            return

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer),
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.vocab_size >= np.iinfo(np.uint16).max:
        raise ValueError("Tokenizer vocabulary does not fit in uint16")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".part.npy")
    token_ids = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.uint16,
        shape=(row_count, args.block_size),
    )
    offset = 0
    if args.parquet:
        import pyarrow.parquet as pq

        batches = (
            batch.to_pandas() for batch in pq.ParquetFile(args.parquet).iter_batches(
                batch_size=args.csv_chunksize,
                columns=["source", "source_index", "processed_func"],
            )
        )
    else:
        batches = pd.read_csv(
            args.csv, usecols=["processed_func"], chunksize=args.csv_chunksize, low_memory=False
        )
    for chunk in batches:
        if args.parquet:
            chunk = chunk[chunk.source == args.source]
            if chunk.empty:
                continue
            if not np.array_equal(chunk.source_index.to_numpy(), np.arange(offset, offset + len(chunk))):
                raise ValueError(f"Unexpected source row order for {args.source}")
        functions = chunk["processed_func"].fillna("").astype(str).tolist()
        for batch_start in range(0, len(functions), args.tokenizer_batch_size):
            batch = functions[
                batch_start : batch_start + args.tokenizer_batch_size
            ]
            encoded = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=args.block_size,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            values = np.asarray(encoded, dtype=np.uint16)
            end = offset + len(values)
            if end > row_count:
                raise ValueError("CSV contains more rows than the manifest")
            token_ids[offset:end] = values
            offset = end
        token_ids.flush()
        print(
            json.dumps(
                {
                    "event": "tokenize_progress",
                    "source": args.source,
                    "rows": offset,
                    "total": row_count,
                }
            ),
            flush=True,
        )
    if offset != row_count:
        raise ValueError(f"CSV rows {offset} do not match manifest rows {row_count}")
    del token_ids
    temporary.replace(args.output)
    metadata = {
        "source": args.source,
        "rows": row_count,
        "block_size": args.block_size,
        "dtype": "uint16",
        "tokenizer": str(args.tokenizer),
        "sha256": sha256_file(args.output),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
