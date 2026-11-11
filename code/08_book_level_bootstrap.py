"""Book-level bootstrap of Cliff's delta (Supplementary Table S2): texts from the same book are resampled together.
In each resample the books of each genre are drawn with replacement (as many draws as the genre has books in the
analysis sample), all texts of a drawn book enter, and Cliff's delta is computed; the interval is the 2.5-97.5
percentile interval of 10,000 resamples (seed 42).
Inputs: outputs/statistics/<encoder>/Pure_Metrics_Data_Final_5D.csv, data/sample_rows.csv, data/corpus_table.csv.
Run from the repository root:
    python code/08_book_level_bootstrap.py > outputs/book_level_bootstrap.txt
"""
import os, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

CSV_OUT = "outputs/book_level_bootstrap.csv"
N_BOOT, SEED = 10000, 42
MET = [("delta", "δ"), ("surprisal_std", "σ_S"), ("global_resonance", "R"), ("curvature", "κ"), ("spec_centroid", "Ω")]


def cliffs_delta(a, b):
    return 2 * mannwhitneyu(a, b, alternative="two-sided").statistic / (len(a) * len(b)) - 1


s = pd.read_csv(os.path.join("data", "sample_rows.csv"))
c = pd.read_csv(os.path.join("data", "corpus_table.csv"))
key = s.merge(c[["genre", "row", "book_title"]], on=["genre", "row"], how="left").rename(columns={"book_title": "book"})
assert len(key) == 758 and key.book.notna().all()
print(f"books in the analysis sample: flash fiction {key[key.genre == 'FlashFiction'].book.nunique()}, prose poetry {key[key.genre == 'ProsePoem'].book.nunique()}")
print(f"{'model':24s} {'measure':8s} {'delta':>8s} {'CI low':>8s} {'CI high':>8s}")
rows = []
for model in ["ko-sroberta-multitask", "bge-m3", "multilingual-e5-large"]:
    m = pd.read_csv(os.path.join("outputs", "statistics", model, "Pure_Metrics_Data_Final_5D.csv"))
    d = m.merge(key.rename(columns={"sample_index": "pkl_index"})[["genre", "pkl_index", "book"]], on=["genre", "pkl_index"], how="left")
    assert d.book.notna().all() and len(d) == 758, "join check failed"
    ff, pp = d[d.is_flash == 1], d[d.is_flash == 0]
    ff_books = [g.index.values for _, g in ff.groupby("book", sort=True)]
    pp_books = [g.index.values for _, g in pp.groupby("book", sort=True)]
    rng = np.random.default_rng(SEED)
    draws = [(np.concatenate([ff_books[i] for i in rng.integers(0, len(ff_books), len(ff_books))]),
              np.concatenate([pp_books[i] for i in rng.integers(0, len(pp_books), len(pp_books))])) for _ in range(N_BOOT)]
    for met, name in MET:
        v = d[met]
        obs = cliffs_delta(ff[met].values, pp[met].values)
        boot = np.array([cliffs_delta(v.loc[a].values, v.loc[b].values) for a, b in draws])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{model:24s} {name:8s} {obs:+8.3f} {lo:+8.3f} {hi:+8.3f}")
        rows.append({"encoder": model, "metric": met, "cliffs_delta": obs, "book_CI_low": lo, "book_CI_high": hi,
                     "n_books_FF": len(ff_books), "n_books_PP": len(pp_books)})
pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
