# -*- coding: utf-8 -*-
"""Composition of the corpus and of the analysis sample.
Reads data/corpus_table.csv and data/sample_rows.csv. Run from the repository root:
    python code/05_sample_composition.py > outputs/sample_composition.txt
"""
import pandas as pd

pd.set_option("display.width", 250)
cnt = pd.read_csv("data/corpus_table.csv", low_memory=False)
sample = pd.read_csv("data/sample_rows.csv")
q = lambda x: f"{x.median():.1f} [{x.quantile(.25):.1f}, {x.quantile(.75):.1f}]"

print("analysis sample:", sample.groupby("genre").size().to_dict())

for genre, bookcol in [("FlashFiction", "book_title"), ("ProsePoem", "book_title")]:
    d = cnt[cnt.genre == genre].copy()
    d["eligible"] = (d.n_sent_source >= 5) & (d.n_sent_traj >= 5)
    d["sample"] = d.row.isin(set(sample[sample.genre == genre].row.astype(int)))
    s = d[d["sample"]]
    print(f"\n== {genre}: corpus {len(d)}, at least five sentences under both splits {int(d.eligible.sum())}, analysis sample {len(s)}")
    print(f"sentences (source-text split) median [IQR]: >= 5 sentences {q(d[d.eligible].n_sent_source)} | sample {q(s.n_sent_source)}")
    print(f"words per sentence median [IQR]:            >= 5 sentences {q(d[d.eligible].wps_source)} | sample {q(s.wps_source)}")
    print(f"trajectory sentences median [IQR], sample: {q(s.n_sent_traj)} | trajectory words per sentence: {q(s.wps_traj)}")
    print(f"words median [IQR], sample: source-text split {q(s.n_sent_source * s.wps_source)} | trajectory {q(s.n_sent_traj * s.wps_traj)}")
    print(f"books: corpus {d[bookcol].nunique()} | sample {s[bookcol].nunique()} ; authors: corpus {d['author'].nunique()} | sample {s['author'].nunique()} ; publication years: corpus {int(pd.to_numeric(d['year']).min())}-{int(pd.to_numeric(d['year']).max())}")
    top = s[bookcol].value_counts()
    print(f"largest book in the sample: {top.index[0]} ({top.iloc[0]} texts, {top.iloc[0] / len(s):.1%})")
    yr = pd.to_numeric(d["year"], errors="coerce")
    d["period"] = pd.cut(yr, [1999, 2004, 2009, 2014, 2019, 2024], labels=["2000-04", "2005-09", "2010-14", "2015-19", "2020-24"])
    t = pd.DataFrame({">= 5 sentences": d[d.eligible].period.value_counts(), "sample": d[d["sample"]].period.value_counts()}).fillna(0).astype(int).sort_index()
    t[">= 5 sentences %"] = (100 * t[">= 5 sentences"] / t[">= 5 sentences"].sum()).round(1); t["sample %"] = (100 * t["sample"] / t["sample"].sum()).round(1)
    print(t.to_string())
    t = pd.DataFrame({">= 5 sentences": d[d.eligible]["publisher"].value_counts(), "sample": d[d["sample"]]["publisher"].value_counts()}).fillna(0).astype(int).sort_values("sample", ascending=False).head(6)
    t["sample %"] = (100 * t["sample"] / len(s)).round(1)
    print(t.to_string())
    if genre == "FlashFiction":
        nc = d.nationality.fillna("(not recorded)").value_counts()
        print(f"authors outside Korea in the corpus: {len(d) - int(nc.get('한국', 0)) - int(nc.get('(not recorded)', 0))} ({(len(d) - int(nc.get('한국', 0)) - int(nc.get('(not recorded)', 0))) / len(d):.1%}), nationality not recorded: {int(nc.get('(not recorded)', 0))}")
        n = s.nationality.fillna("(not recorded)").value_counts()
        print("author nationality in the sample:", n.to_dict())
        print(f"authors outside Korea in the sample: {len(s) - int(n.get('한국', 0)) - int(n.get('(not recorded)', 0))} ({(len(s) - int(n.get('한국', 0)) - int(n.get('(not recorded)', 0))) / len(s):.1%})")
