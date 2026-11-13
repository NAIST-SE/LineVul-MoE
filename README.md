# LineVul-MoE reproduction

This repository contains the code for two BigVul splits, 11 model variants, and
five training seeds. The 110 checkpoints are available at
[Hietan/LineVul-MoE](https://huggingface.co/Hietan/LineVul-MoE).

| Directory | Purpose |
| --- | --- |
| data/ | Prepare the official and project splits, token caches, and line labels |
| training/ | Train All-in-One, five Exclude experts, and five One-Hot experts per split and seed |
| inference/ | Save function predictions and line attention scores |
| evaluation/ | Aggregate the experiment tables |
| core/ | Shared model architecture and helpers |

## Inputs and environment

Use Python 3.10 on Linux and install the pinned dependencies with
`python -m pip install -r requirements.txt`. Training uses a CUDA GPU with BF16.
The data stage downloads `data/functions.parquet`, the two original split
manifests under `prepared/`, and the test line labels under `line-data/`
from [Hietan/LineVul-MoE_Data](https://huggingface.co/datasets/Hietan/LineVul-MoE_Data).
The Parquet stores each function once per original source row. Both splits use
the same function table and preserve `source` and `source_index`.
For local preparation, pass `--raw /path/to/csvs` with the three original files
from the [LineVul dataset instructions](https://github.com/awsm-research/LineVul#dataset).
The local path verifies the original file hashes and reconstructs both splits.

Download a local snapshot of
[microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base) with its
tokenizer and PyTorch weights. The original files' hashes are available in the
[release metadata](https://huggingface.co/Hietan/LineVul-MoE/blob/92a0840b31f871e3020c9838f6e9f44ead365163/reproduction.json).
Raw CSVs, checkpoints, and generated outputs belong outside tracked files.

## Run

```bash
python reproduce.py data --base-model /path/to/codebert-base

# Reuse the published weights at the pinned Hub revision.
python reproduce.py infer --weights hub --base-model /path/to/codebert-base
python reproduce.py evaluate

# Retrain, then infer with those checkpoints instead.
python reproduce.py data --artifacts retrained --base-model /path/to/codebert-base
python reproduce.py train --artifacts retrained --base-model /path/to/codebert-base
python reproduce.py infer --artifacts retrained --weights local --base-model /path/to/codebert-base
python reproduce.py evaluate --artifacts retrained
```

For the paper release, pass `--data-revision 7f886abf80fd384ee1b2bc08eb0e60682cac1c8e` to the data
stage. The archived dataset revision is `7f886abf80fd384ee1b2bc08eb0e60682cac1c8e`. Access to the private dataset repository is required.

Use `--split official --seed 42` to select one 11-model group in the training,
inference, or evaluation stages. Use `--artifacts PATH` to set the work directory
(default `artifacts/`). Training is sequential. Each One-Hot model starts from
the same split and seed's All-in-One checkpoint. Use
`python reproduce.py train --dry-run` to list the selected checkpoint IDs.
Hardware and software differences can affect retraining results.

The data stage preserves the official split and groups related projects, CVEs,
commits, and duplicate functions for the project split (split seed 20260729).
The five training seeds use the same fixed split. Inference emits
`predictions.csv.gz` and `line_scores.csv.gz` per model. The line scores retain
the original first-layer attention aggregation, including padded query positions.

## Evaluation tables

Results under `artifacts/results/` are split-specific. Each table has a
per-seed CSV and a seed-mean CSV.

- `line-by-cwe` has line AP, Top-1, and Top-10 by model and evaluation CWE.
- `line-routing` compares All-in-One, maximum line-score selection, and oracle
  selection. Each router uses one expert's complete ranking per function.
- `line-agreement` has cosine similarity, Spearman correlation, and first-ranked
  line agreement between experts of the same family and between G1 and each expert.
- `binary-by-model`, `binary-cross-cwe`, and `binary-by-cve` have Accuracy, F1,
  Recall, and Precision. The cross-CWE table holds the test set fixed across
  models within a row. The One-Hot model-level table uses its conditional task.

Line metrics use Top-5 vulnerable functions with a verified flaw line within the
512-token input. Routing metrics weight functions equally. Top-k and first-ranked
line ties use ascending line index. Undefined metrics are left blank.

## Aggregate the archived experiment outputs

Once the dataset repository is populated, download the original predictions and
evaluate them without downloading the model weights or running inference.

```bash
python reproduce.py fetch-outputs --data-revision 7f886abf80fd384ee1b2bc08eb0e60682cac1c8e
python reproduce.py evaluate
```

Each of the 110 model folders provides `predictions.csv.gz` and
`line_scores.csv.gz`. The downloader verifies the dataset manifest checksums
and links to the Hub cache. Keep that cache available when running evaluation.
