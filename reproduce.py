"""Run the data, training, inference, and evaluation stages."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SPLITS = ("official", "project")
SEEDS = (123456, 42, 2026, 31415, 27182)
CWES = (119, 20, 399, 125, 200)
HUB = "Hietan/LineVul-MoE"
DATASET = "Hietan/LineVul-MoE_Data"
RAW_SHA256 = {
    "train": "a73a4a177569ff71038f43040c5301ed9d41cd61ca62f13e772de354e124d381",
    "valid": "74cf46e16107e6f3d8692279489104b01985aece18be2145e21eff2f75ce391d",
    "test": "e1ac10e2a05b57d01ee3b180dffa84284deb435448e79332d12a50bad85b9832",
}
REVISION = "92a0840b31f871e3020c9838f6e9f44ead365163"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw(directory):
    for source, expected in RAW_SHA256.items():
        path = directory / f"{source}.csv"
        if sha256(path) != expected:
            raise ValueError(f"Unexpected SHA-256 for {path}. Use the original LineVul CSV.")


def fetch_data(out, revision):
    from huggingface_hub import hf_hub_download

    def download(name):
        return Path(hf_hub_download(DATASET, name, repo_type="dataset", revision=revision))

    manifest = download("files.csv")
    revision = manifest.parent.name
    with manifest.open(newline="") as handle:
        hashes = {row["path"]: row["sha256"] for row in csv.DictReader(handle)}
    names = ["data/functions.parquet"]
    for split in SPLITS:
        names.extend(f"prepared/{split}/{name}" for name in ("manifest.csv.gz", "top5.json", "audit.json"))
        names.append(f"line-data/{split}.csv.gz")
    for name in names:
        cached = download(name)
        if sha256(cached) != hashes[name]:
            raise ValueError(f"Unexpected SHA-256 for {name}")
        if name == names[0]:
            continue
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(target) != hashes[name]:
                raise FileExistsError(f"Use a separate --artifacts directory: {target}")
        else:
            target.symlink_to(cached.resolve())
    return download(names[0])


def fetch_outputs(out, splits, seeds, revision):
    from huggingface_hub import hf_hub_download

    def download(name):
        return Path(hf_hub_download(DATASET, name, repo_type="dataset", revision=revision))

    manifest = download("files.csv")
    # Resolve main once so a concurrent publication cannot mix snapshots.
    revision = manifest.parent.name
    with manifest.open(newline="") as handle:
        hashes = {row["path"]: row["sha256"] for row in csv.DictReader(handle)}
    for split in splits:
        for seed in seeds:
            for _, _, model in models(split, seed):
                for filename in ("predictions.csv.gz", "line_scores.csv.gz"):
                    relative = f"outputs/{model}/{filename}"
                    cached = download(relative)
                    if sha256(cached) != hashes[relative]:
                        raise ValueError(f"Unexpected SHA-256 for {relative}")
                    target = out / "inference" / model / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        if sha256(target) != hashes[relative]:
                            raise FileExistsError(f"Use a separate --artifacts directory: {target}")
                        continue
                    target.symlink_to(cached.resolve())


def models(split, seed):
    yield "All-in-One", None, f"LineVul-MoE_All-in-One_{split}_seed{seed}"
    for method in ("Exclude", "One-Hot"):
        for i, cwe in enumerate(CWES):
            yield method, i, f"LineVul-MoE_{method}_{split}_CWE-{cwe}_seed{seed}"


def run(script, *args):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "core"), TOKENIZERS_PARALLELISM="false")
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("data", "train", "infer", "evaluate", "fetch-outputs"))
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--raw", type=Path, help="Use local train.csv, valid.csv and test.csv instead of the Hub dataset")
    parser.add_argument("--data-revision", default="main", help="Hub dataset revision (use a commit SHA to reproduce a release)")
    parser.add_argument("--base-model", type=Path, help="Local CodeBERT snapshot")
    parser.add_argument("--split", choices=SPLITS, action="append")
    parser.add_argument("--seed", choices=SEEDS, type=int, action="append")
    parser.add_argument("--weights", choices=("hub", "local"), default="hub")
    parser.add_argument("--dry-run", action="store_true", help="List selected checkpoint names")
    args = parser.parse_args()
    out = args.artifacts.resolve()
    prepared, cache = out / "prepared", out / "cache"
    splits = tuple(dict.fromkeys(args.split or SPLITS))
    seeds = tuple(dict.fromkeys(args.seed or SEEDS))
    if args.dry_run:
        for split in splits:
            for seed in seeds:
                for _, _, name in models(split, seed):
                    print(name)
        return
    if args.stage == "fetch-outputs":
        fetch_outputs(out, splits, seeds, args.data_revision)
        return
    if args.stage != "evaluate" and args.base_model is None:
        parser.error("--base-model is required")
    if args.stage == "data":
        if args.raw is None:
            parquet = fetch_data(out, args.data_revision)
        else:
            verify_raw(args.raw)
            run("data/prepare_data.py", "--data-dir", args.raw, "--output-dir", prepared)
            for split in SPLITS:
                run("data/prepare_line_data.py", "--manifest", prepared / split / "manifest.csv.gz",
                    "--raw-data-dir", args.raw, "--output", out / "line-data" / f"{split}.csv.gz")
        for source in RAW_SHA256:
            input_args = ("--parquet", parquet) if args.raw is None else ("--csv", args.raw / f"{source}.csv")
            run("data/tokenize_data.py", *input_args,
                "--manifest", prepared / "official/manifest.csv.gz", "--source", source,
                "--tokenizer", args.base_model, "--output", cache / f"{source}.input_ids.npy")
        return
    if args.stage == "evaluate":
        for split in splits:
            run("evaluation/paper.py", "--input", out / "inference", "--output", out / "results",
                "--split", split, "--seeds", *seeds)
        return
    calibration = {}
    if args.stage == "infer" and args.weights == "hub":
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HUB, "reproduction.json", revision=REVISION)
        metadata = json.loads(Path(path).read_text())
        calibration = metadata["inference_calibration"]
        for filename, expected in metadata["base_model_sha256"].items():
            if sha256(args.base_model / filename) != expected:
                raise ValueError(f"Unexpected SHA-256 for CodeBERT file {filename}")
        manifest = hf_hub_download(HUB, "manifest.csv", revision=REVISION)
        with open(manifest, newline="") as handle:
            expected_weights = {row["model_name"]: row for row in csv.DictReader(handle)}
    for split in splits:
        for seed in seeds:
            group = list(models(split, seed))
            for method, expert, name in group:
                model_dir = out / "training" / name
                if args.stage == "train":
                    if (model_dir / "run.json").is_file() and (model_dir / "best.pt").is_file():
                        print(f"Already trained: {name}", flush=True)
                        continue
                    options = []
                    if expert is not None:
                        options += ["--expert-index", expert]
                    if method == "One-Hot":
                        options += ["--warm-start", out / "training" / group[0][2] / "best.pt"]
                    run("training/train.py", "--model-kind", "generalist" if expert is None else "specialist",
                        "--manifest", prepared / split / "manifest.csv.gz", "--top5", prepared / split / "top5.json",
                        "--model-path", args.base_model, "--cache-dir", cache, "--output-dir", model_dir,
                        "--seed", seed, "--train-batch-size", 64, "--eval-batch-size", 128, "--precision", "bf16",
                        "--expert-negative-mode", "conditional-one-hot" if method == "One-Hot" else "exclude",
                        "--imbalance", "class_balanced" if method == "One-Hot" else "bce", *options)
                    continue
                if args.weights == "hub":
                    weights = hf_hub_download(HUB, "model.safetensors", subfolder=f"checkpoints/{name}",
                                              revision=REVISION)
                    if sha256(Path(weights)) != expected_weights[name]["sha256"]:
                        raise ValueError(f"Unexpected SHA-256 for {name}")
                    settings = calibration[name]
                else:
                    weights = model_dir / "best.pt"
                    settings = json.loads((model_dir / "run.json").read_text())["calibration"]
                inference_dir = out / "inference" / name
                if any((inference_dir / name).is_symlink() for name in ("predictions.csv.gz", "line_scores.csv.gz")):
                    raise FileExistsError(f"Archived predictions in {inference_dir}. Use a separate --artifacts directory.")
                run("inference/evaluate_line_level.py", "--weights", weights, "--temperature", settings["temperature"], "--threshold", settings["threshold"],
                    "--seed", seed, "--line-data", out / "line-data" / f"{split}.csv.gz",
                    "--cache-dir", cache, "--model-path", args.base_model, "--protocol", split,
                    "--output-dir", inference_dir)


if __name__ == "__main__":
    main()
