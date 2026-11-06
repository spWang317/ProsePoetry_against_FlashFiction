# -*- coding: utf-8 -*-
"""Length matching of the two genres.

Texts with at least five sentences under both splits (source-text split and trajectory split) are placed in a
grid of source-text sentence count (2-sentence steps) by words per sentence (1.5-word steps). In every cell
that holds at least two texts of each genre, the same number of texts is drawn from both genres.
Run from the repository root:
    python code/01_length_matching.py
Reads data/corpus_table.csv. Writes data/sample_rows.csv
(genre, sample_index, row, n_sentences, tokens_per_sent).
"""
import numpy as np, pandas as pd


def stratified_match(df_a, df_b, seed=42, s_step=2, d_step=1.5, min_per_bin=2):
    rng = np.random.default_rng(seed)
    a, b = df_a.copy(), df_b.copy()
    
    s_min = min(a["n_sentences"].min(), b["n_sentences"].min())
    s_max = max(a["n_sentences"].max(), b["n_sentences"].max())
    d_min = min(a["tokens_per_sent"].min(), b["tokens_per_sent"].min())
    d_max = max(a["tokens_per_sent"].max(), b["tokens_per_sent"].max())

    s_bins = np.arange(np.floor(s_min), np.ceil(s_max) + s_step, s_step)
    d_bins = np.arange(np.floor(d_min), np.ceil(d_max) + d_step, d_step)

    for df in [a, b]:
        df["s_bin"] = pd.cut(df["n_sentences"], bins=s_bins, include_lowest=True)
        df["d_bin"] = pd.cut(df["tokens_per_sent"], bins=d_bins, include_lowest=True)
        df["bin_key"] = df["s_bin"].astype(str) + "|" + df["d_bin"].astype(str)

    common_bins = sorted(set(a["bin_key"].unique()) & set(b["bin_key"].unique()))
    a_out, b_out = [], []
    for k in common_bins:
        sub_a, sub_b = a[a["bin_key"] == k], b[b["bin_key"] == k]
        n = min(len(sub_a), len(sub_b))
        if n < min_per_bin: continue
        a_out.append(sub_a.sample(n=n, random_state=seed))
        b_out.append(sub_b.sample(n=n, random_state=seed))
    
    if not a_out: raise ValueError("Matching failed. No common bins found.")
    return pd.concat(a_out, ignore_index=True), pd.concat(b_out, ignore_index=True)


corpus = pd.read_csv("data/corpus_table.csv", low_memory=False)
pool = {}
for genre in ["FlashFiction", "ProsePoem"]:
    d = corpus[corpus.genre == genre].reset_index(drop=True)
    d["n_sentences"], d["tokens_per_sent"] = d["n_sent_source"], d["wps_source"]
    pool[genre] = d[(d.n_sent_source >= 5) & (d.n_sent_traj >= 5)].copy()
    print(f"{genre}: corpus {len(d)} | at least five sentences under both splits {len(pool[genre])}")
m_ff, m_pp = stratified_match(pool["FlashFiction"], pool["ProsePoem"])


def cells(a, b, s_step=2, d_step=1.5):
    s_bins = np.arange(np.floor(min(a.n_sentences.min(), b.n_sentences.min())), np.ceil(max(a.n_sentences.max(), b.n_sentences.max())) + s_step, s_step)
    d_bins = np.arange(np.floor(min(a.tokens_per_sent.min(), b.tokens_per_sent.min())), np.ceil(max(a.tokens_per_sent.max(), b.tokens_per_sent.max())) + d_step, d_step)
    return [pd.cut(d.n_sentences, bins=s_bins, include_lowest=True).astype(str) + "|" + pd.cut(d.tokens_per_sent, bins=d_bins, include_lowest=True).astype(str) for d in (a, b)]


ka, kb = cells(pool["FlashFiction"], pool["ProsePoem"])
ca, cb = ka.value_counts(), kb.value_counts()
shared = sorted(set(ca.index) & set(cb.index))
used = [k for k in shared if min(ca[k], cb[k]) >= 2]
print(f"cells shared by both genres: {len(shared)} (FF {int(ka.isin(shared).sum())}, PP {int(kb.isin(shared).sum())}) | "
      f"with at least two texts per genre: {len(used)} (FF {int(ka.isin(used).sum())}, PP {int(kb.isin(used).sum())})")
print(f"matched: FF {len(m_ff)}, PP {len(m_pp)}")
rows = []
for genre, m in [("FlashFiction", m_ff), ("ProsePoem", m_pp)]:
    for k, r in enumerate(m.itertuples()):
        rows.append({"genre": genre, "sample_index": k, "row": int(r.row), "n_sentences": int(r.n_sentences), "tokens_per_sent": float(r.tokens_per_sent)})
pd.DataFrame(rows).to_csv("data/sample_rows.csv", index=False)
