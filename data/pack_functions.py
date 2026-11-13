"""Store the shared function bodies once, preserving original source row IDs."""
import argparse
import csv
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduce import RAW_SHA256, verify_raw

SCHEMA = pa.schema([
    ("source", pa.string()), ("source_index", pa.int64()),
    ("processed_func", pa.string()), ("target", pa.int8()),
    ("flaw_line_index", pa.string()),
])


def pack(raw, output):
    verify_raw(raw)
    csv.field_size_limit(sys.maxsize)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".partial")
    counts = {}
    with pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as writer:
        for source in RAW_SHA256:
            with (raw / f"{source}.csv").open(encoding="utf-8", newline="") as handle:
                rows = []
                for index, row in enumerate(csv.DictReader(handle)):
                    rows.append(dict(source=source, source_index=index,
                                     processed_func=row["processed_func"], target=int(row["target"]),
                                     flaw_line_index=row["flaw_line_index"]))
                    if len(rows) == 1000:
                        writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
                        rows.clear()
                if rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
                counts[source] = index + 1
    with pq.ParquetFile(tmp) as parquet:
        assert parquet.metadata.num_rows == sum(counts.values())
        for source in counts:
            with (raw / f"{source}.csv").open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                index = 0
                for batch in parquet.iter_batches(batch_size=1000):
                    for row in batch.to_pylist():
                        if row["source"] == source:
                            original = next(reader)
                            assert row["source_index"] == index
                            assert row["processed_func"] == original["processed_func"]
                            assert row["target"] == int(original["target"])
                            assert row["flaw_line_index"] == original["flaw_line_index"]
                            index += 1
                assert index == counts[source] and next(reader, None) is None
    tmp.replace(output)
    print(counts, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    pack(args.raw, args.output)
