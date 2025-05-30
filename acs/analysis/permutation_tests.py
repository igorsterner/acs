import argparse
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = "data/results"


def permutation_test(x, y, num_rounds=10000, seed=None):
    rng = np.random.RandomState(seed)
    m, n = len(x), len(y)
    if m != n:
        print(x)
        print(y)
        raise ValueError("x and y must have the same length if `paired=True`")
    sample_x = np.empty(m)
    sample_y = np.empty(n)
    at_least_as_extreme = 0.0
    reference_stat = np.abs(np.mean(x) - np.mean(y))
    for i in range(num_rounds):
        flip = rng.randn(m) > 0.0
        for idx, f in enumerate(flip):
            if f:
                sample_x[idx], sample_y[idx] = y[idx], x[idx]
            else:
                sample_x[idx], sample_y[idx] = x[idx], y[idx]
        diff = np.abs(np.mean(sample_x) - np.mean(sample_y))
        if diff > reference_stat or np.isclose(diff, reference_stat):
            at_least_as_extreme += 1.0
    return at_least_as_extreme / num_rounds


def load_scores_jsonl(filepath):
    scores = []
    with open(filepath, "r") as fin:
        for line in fin:
            item = json.loads(line)
            observed = item.get("observed_logprob")
            manipulated = item.get("manipulated_logprob")
            if observed is not None and manipulated is not None:
                scores.append(1 if observed > manipulated else 0)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", required=True)
    parser.add_argument("--lms", nargs="+", required=True)
    args = parser.parse_args()

    for lang in args.langs:
        all_scores = {}
        lang_dir = os.path.join(DATA_DIR, lang)

        for lm in args.lms:
            lm_file = lang_dir / f"{lm}.jsonl"
            if not lm_file.exists():
                print(f"File not found: {lm_file}")
                continue
            scores = load_scores_jsonl(lm_file)
            if len(scores) == 0:
                print(f"No usable scores found in {lm_file}, skipping.")
                continue
            all_scores[lm] = scores

        systems = list(all_scores.keys())
        if not systems:
            print(f"No systems found for {lang}, skipping.")
            continue
        systems = sorted(systems, key=lambda x: np.mean(all_scores[x]))
        num_systems = len(systems)

        pvalues = pd.DataFrame(np.nan, index=systems, columns=systems)
        total_permutation_tests = num_systems * (num_systems - 1) // 2

        pbar = tqdm(total=total_permutation_tests, desc=f"Permutation ({lang})")
        for i in range(num_systems):
            for j in range(i + 1, num_systems):
                s_i, s_j = systems[i], systems[j]
                pval = permutation_test(
                    all_scores[s_i], all_scores[s_j], num_rounds=10000
                )
                pvalues.at[s_i, s_j] = pval
                pbar.update(1)
        pbar.close()

        print(lang)
        print(pvalues.to_string(na_rep="   -"))
        print("\nMeans for each system:")
        for system in systems:
            avg = np.mean(all_scores[system])
            print(f"{system}: {avg*100:.2f}")


if __name__ == "__main__":
    main()
