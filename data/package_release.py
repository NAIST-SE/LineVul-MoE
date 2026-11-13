"""Stage the original experiment inputs and archived model outputs for the Hub."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduce import CWES, REVISION, SEEDS, SPLITS, models, sha256
from pack_functions import pack


def package(experiment, raw, destination):
    sources = {}
    for split in SPLITS:
        for filename in ("manifest.csv.gz", "top5.json", "audit.json"):
            path = f"prepared/{split}/{filename}"
            sources[path] = experiment / "data" / path
        path = f"line-data/{split}.csv.gz"
        sources[path] = experiment / "data/line-level" / f"{split}.csv.gz"
        for seed in SEEDS:
            for method, index, name in models(split, seed):
                one_hot = method == "One-Hot"
                if index is None:
                    run_name = f"generalist-seed{seed}-frac1.0-bce"
                    expert = "g1"
                else:
                    suffix = "class_balanced" if one_hot else "bce"
                    mode = "conditional-one-hot" if one_hot else "exclude"
                    run_name = f"specialist-seed{seed}-frac1.0-{suffix}-expert{index}-{mode}"
                    expert = f"expert{index}"
                series = "line-one-hot" if one_hot else "line-specialization"
                run = experiment / "runs" / split / run_name
                line = experiment / "runs" / series / split / f"seed{seed}" / expert
                metadata = json.loads((line / "run.json").read_text())
                if (metadata["protocol"], metadata["seed"], metadata["expert_index"]) != (split, seed, index):
                    raise ValueError(f"Line output metadata does not match {name}")
                sources[f"outputs/{name}/line_scores.csv.gz"] = line / "line_scores.csv.gz"
                sources[f"outputs/{name}/predictions.csv.gz"] = run / "predictions.csv.gz"
    if not all(path.is_file() for path in sources.values()):
        raise FileNotFoundError([str(path) for path in sources.values() if not path.is_file()])
    destination.mkdir(parents=True, exist_ok=False)
    functions = destination / "data/functions.parquet"
    pack(raw, functions)
    records = [dict(path="data/functions.parquet", sha256=sha256(functions), bytes=functions.stat().st_size)]
    for relative, path in sources.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        digest = sha256(path)
        if sha256(target) != digest:
            raise ValueError(f"Checksum mismatch: {relative}")
        records.append(dict(path=relative, sha256=digest, bytes=target.stat().st_size))
    with (destination / "files.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "sha256", "bytes"))
        writer.writeheader()
        writer.writerows(records)
    (destination / "README.md").write_text(
        "# LineVul-MoE Data\n\n"
        "Research artifacts derived from [LineVul](https://github.com/awsm-research/LineVul) "
        "and [BigVul](https://doi.org/10.1145/3379597.3387501).\n\n"
        "- The shared function table in `data/functions.parquet` retains the original LineVul source and row index.\n"
        "- `prepared/` maps source rows to the official and project partitions.\n"
        "- `line-data/` contains test function metadata and reference line labels.\n"
        "- `outputs/` contains the archived function and line predictions for all 110 checkpoints.\n"
        "- `files.csv` lists the SHA-256 and size of each data file.\n\n"
        "BigVul and LineVul upstream repositories use MIT licenses. Function excerpts retain their original project terms.\n\n"
        f"Checkpoints: [Hietan/LineVul-MoE](https://huggingface.co/Hietan/LineVul-MoE/tree/{REVISION}).\n"
        f"CWEs: {', '.join(map(str, CWES))}. Each partition has 55 checkpoints.\n\n"
        "To aggregate the archived predictions in the replication package, run "
        "`python reproduce.py fetch-outputs --data-revision <dataset-commit>`, "
        "then `python reproduce.py evaluate`.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    package(args.experiment, args.raw, args.destination)
