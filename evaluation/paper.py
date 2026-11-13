"""Aggregate function detection, line localization, routing and agreement."""
import argparse
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduce import CWES, models
from analyze_line_specialization import function_metrics


def binary(frame):
    y = frame.target.to_numpy(dtype=bool)
    p = frame.prediction.to_numpy(dtype=bool)
    tp = int((y & p).sum())
    def divide(a, b):
        return a / b if b else np.nan
    return dict(n=len(y), positives=int(y.sum()), accuracy=np.mean(y == p),
                recall=divide(tp, y.sum()), precision=divide(tp, p.sum()),
                f1=divide(2 * tp, y.sum() + p.sum()))


def compare(a, b):
    scores = []
    for x, y in zip(a, b, strict=True):
        norm = np.linalg.norm(x) * np.linalg.norm(y)
        rank = spearmanr(x, y).statistic if len(x) > 1 and np.ptp(x) and np.ptp(y) else np.nan
        scores.append((np.dot(x, y) / norm if norm else np.nan, rank, np.argmax(x) == np.argmax(y)))
    return dict(zip(("cosine", "spearman", "top1_agreement"),
                    pd.DataFrame(scores).mean()))


def summarize(records, keys, output):
    frame = pd.DataFrame(records)
    frame.to_csv(output.with_name(output.stem + "-per-seed.csv"), index=False)
    means = frame.drop(columns="seed").groupby(keys, dropna=False, sort=False).mean(numeric_only=True)
    counts = frame.groupby(keys, dropna=False, sort=False).count()
    for name in ("accuracy", "f1", "recall", "precision"):
        if name in means:
            means[name + "_seeds"] = counts[name]
    means.to_csv(output)


def evaluate(input_dir, output, split, seeds):
    line, route, pairs, functions, cross, cves = [], [], [], [], [], []
    fields = ["line_ap", "top1", "top10"]

    def routing_row(seed, family, routing, selected, labels):
        macro = selected.assign(eval_cwe=labels.to_numpy()).groupby("eval_cwe")[fields].mean().mean()
        return dict(seed=seed, method=family, routing=routing, **selected[fields].mean().to_dict(),
                    **{f"macro_{key}": value for key, value in macro.items()})

    for seed in seeds:
        specs = list(models(split, seed))
        frames, scores, decisions = {}, {}, {}
        for method, index, name in specs:
            folder = input_dir / name
            path = folder / "line_scores.csv.gz"
            frames[name] = function_metrics(path).set_index("row_id")
            scores[name] = pd.read_csv(path, usecols=["row_id", "line_index", "score", "label"])
            decisions[name] = pd.read_csv(folder / "predictions.csv.gz").set_index("row_id", verify_integrity=True)
        g1 = specs[0][2]
        reference = frames[g1]
        ids = reference.index[reference.verified_flaws > 0]
        if not len(ids):
            raise ValueError("No Top-5 vulnerable functions have verified lines in the input window")
        labels = reference.loc[ids, "cwe_index"]
        arrays, maxima = {}, {}
        for method, index, name in specs:
            f = frames[name].loc[ids]
            assert f.cwe_index.equals(labels) and (f.verified_flaws > 0).all()
            for c, group in f.groupby("cwe_index"):
                line.append(dict(seed=seed, model=name.rsplit("_seed", 1)[0],
                                 eval_cwe=CWES[c], **group[fields].mean().to_dict()))
            groups = scores[name].loc[scores[name].row_id.isin(ids)].sort_values("line_index").groupby("row_id")
            aligned = [groups.get_group(key) for key in ids]
            keys = [a[["line_index", "label"]].to_numpy() for a in aligned]
            if name == g1:
                reference_keys = keys
            else:
                assert all(np.array_equal(a, b) for a, b in zip(reference_keys, keys, strict=True))
            arrays[name] = [a.score.to_numpy() for a in aligned]
            if not all(np.isfinite(a).all() for a in arrays[name]):
                raise ValueError(f"Non-finite line scores for {name}")
            maxima[name] = np.array([a.max() for a in arrays[name]])

            if set(decisions[name].index) != set(decisions[g1].index):
                raise ValueError(f"Test functions differ for {name}")
            d = decisions[name].loc[decisions[g1].index]
            assert d[["target", "cwe_index"]].equals(decisions[g1][["target", "cwe_index"]])
            short = name.rsplit("_seed", 1)[0]
            for j, cwe in enumerate(CWES):
                population = d[(d.target == 0) | (d.cwe_index == j)]
                cross.append(dict(seed=seed, model=short, eval_cwe=cwe, **binary(population)))
            if method == "One-Hot":
                d = d[(d.target == 1) & d.cwe_index.between(0, 4)].copy()
                d["target"] = (d.cwe_index == index).astype(int)
            elif index is not None:
                d = d[(d.target == 0) | (d.cwe_index == index)]
            functions.append(dict(seed=seed, model=short, **binary(d)))
            for cve, group in d.groupby(d.cve_id.fillna("__MISSING_CVE__")):
                cves.append(dict(seed=seed, model=short, cve=cve, **binary(group)))

        for family in ("Exclude", "One-Hot"):
            names = [name for method, _, name in specs if method == family]
            best = np.argmax(np.stack([maxima[name] for name in names]), axis=0)
            cube = np.stack([frames[name].loc[ids, fields].to_numpy() for name in names])
            for label, chosen in (("maximum-line-score", best), ("oracle", labels.to_numpy())):
                selected = pd.DataFrame(cube[chosen, np.arange(len(ids))], columns=fields)
                route.append(routing_row(seed, family, label, selected, labels))
            for a, b in list(combinations(names, 2)) + [(g1, name) for name in names]:
                pairs.append(dict(seed=seed, a=a.rsplit("_seed", 1)[0], b=b.rsplit("_seed", 1)[0],
                                  **compare(arrays[a], arrays[b])))
        route.append(routing_row(seed, "All-in-One", "single", reference.loc[ids, fields], labels))

    output.mkdir(parents=True, exist_ok=True)
    for name, records, keys in (
        ("line-by-cwe", line, ["model", "eval_cwe"]),
        ("line-routing", route, ["method", "routing"]),
        ("line-agreement", pairs, ["a", "b"]),
        ("binary-by-model", functions, ["model"]),
        ("binary-cross-cwe", cross, ["model", "eval_cwe"]),
        ("binary-by-cve", cves, ["model", "cve"]),
    ):
        summarize(records, keys, output / f"{split}-{name}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("official", "project"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    evaluate(args.input, args.output, args.split, args.seeds)
