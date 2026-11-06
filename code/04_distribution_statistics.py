# -*- coding: utf-8 -*-
"""Extended distributional statistics of the five measures (Supplementary Information).

Reads the per-document measures written by measures_and_statistics.py
and reports, for each metric:
  - interquartile range within each genre,
  - median difference (Flash Fiction - Prose Poetry) with a percentile bootstrap
    95% confidence interval (documents resampled with replacement within genre),
  - skewness and Pearson kurtosis (normal distribution = 3) of the pooled sample.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

DATA_DIR = "outputs/statistics/ko-sroberta-multitask"
METRICS = ["delta", "surprisal_std", "global_resonance", "curvature", "spec_centroid"]
N_BOOT = 10000
SEED = 42


def iqr(x):
    q1, q3 = np.percentile(x, [25, 75])
    return q3 - q1


def bootstrap_median_diff(a, b, rng, n_boot=N_BOOT):
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = (np.median(rng.choice(a, size=len(a), replace=True))
                    - np.median(rng.choice(b, size=len(b), replace=True)))
    return np.percentile(diffs, [2.5, 97.5])


def main(data_dir=DATA_DIR):
    df = pd.read_csv(os.path.join(data_dir, "Pure_Metrics_Data_Final_5D.csv"))
    rng = np.random.default_rng(SEED)
    rows = []
    for met in METRICS:
        ff = df.loc[df["genre"] == "FlashFiction", met].values
        pp = df.loc[df["genre"] == "ProsePoem", met].values
        lo, hi = bootstrap_median_diff(ff, pp, rng)
        rows.append({
            "Metric": met,
            "FF_IQR": iqr(ff),
            "PP_IQR": iqr(pp),
            "Median_Diff_FF_minus_PP": np.median(ff) - np.median(pp),
            "Boot_CI_Lower": lo,
            "Boot_CI_Upper": hi,
            "Skewness_All": skew(df[met].values),
            "Kurtosis_All": kurtosis(df[met].values, fisher=False),
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(data_dir, "Distribution_Statistics.csv"), index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DATA_DIR, help="folder holding Pure_Metrics_Data_Final_5D.csv")
    main(ap.parse_args().data_dir)
