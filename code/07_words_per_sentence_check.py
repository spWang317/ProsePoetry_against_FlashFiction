"""Sensitivity check (Supplementary Table S7): the genre coefficient of each measure when the trajectory words
per sentence are added as a second covariate.  Model A (the article): measure ~ genre + log(1 + T).
Model B: measure ~ genre + log(1 + T) + trajectory words per sentence.  HC3 standard errors, as in the article.
Inputs: outputs/statistics/<encoder>/Pure_Metrics_Data_Final_5D.csv, data/sample_rows.csv, data/corpus_table.csv.
Run from the repository root:
    python code/07_words_per_sentence_check.py > outputs/words_per_sentence_check.txt
"""
import os, pandas as pd, statsmodels.formula.api as smf

CSV_OUT = "outputs/words_per_sentence_check.csv"
rows = []
s = pd.read_csv(os.path.join("data", "sample_rows.csv"))
c = pd.read_csv(os.path.join("data", "corpus_table.csv"))
key = s.merge(c[["genre", "row", "wps_traj", "n_sent_traj"]], on=["genre", "row"], how="left")
assert key.wps_traj.notna().all() and len(key) == 758
MET = [("delta", "δ"), ("surprisal_std", "σ_S"), ("global_resonance", "R"), ("curvature", "κ"), ("spec_centroid", "Ω")]
print("genre coefficient (flash fiction = 1), HC3; A = + log(1+T) (the article), B = + log(1+T) + trajectory words per sentence")
print(f"{'model':24s} {'measure':8s} {'beta_A':>9s} {'p_A':>7s} {'beta_B':>9s} {'p_B':>7s} {'wps coef':>9s} {'p_wps':>7s}")
for model in ["ko-sroberta-multitask", "bge-m3", "multilingual-e5-large"]:
    m = pd.read_csv(os.path.join("outputs", "statistics", model, "Pure_Metrics_Data_Final_5D.csv"))
    d = m.merge(key.rename(columns={"sample_index": "pkl_index"})[["genre", "pkl_index", "wps_traj", "n_sent_traj"]], on=["genre", "pkl_index"], how="left")
    assert d.wps_traj.notna().all() and (d.n_sent == d.n_sent_traj).all(), "join check failed"
    for met, name in MET:
        a = smf.ols(f"{met} ~ is_flash + log_n_sent", data=d).fit(cov_type="HC3")
        b = smf.ols(f"{met} ~ is_flash + log_n_sent + wps_traj", data=d).fit(cov_type="HC3")
        print(f"{model:24s} {name:8s} {a.params['is_flash']:+9.4f} {a.pvalues['is_flash']:7.3f} {b.params['is_flash']:+9.4f} {b.pvalues['is_flash']:7.3f} {b.params['wps_traj']:+9.4f} {b.pvalues['wps_traj']:7.3f}")
        rows.append({"encoder": model, "metric": met, "beta_A": a.params["is_flash"], "CI_A_low": a.conf_int().loc["is_flash", 0], "CI_A_high": a.conf_int().loc["is_flash", 1], "p_A": a.pvalues["is_flash"],
                     "beta_B": b.params["is_flash"], "CI_B_low": b.conf_int().loc["is_flash", 0], "CI_B_high": b.conf_int().loc["is_flash", 1], "p_B": b.pvalues["is_flash"],
                     "wps_coef": b.params["wps_traj"], "wps_p": b.pvalues["wps_traj"]})
pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
print("\nSpearman correlation of wps_traj with each measure (primary model), and genre medians of wps_traj:")
m = pd.read_csv(os.path.join("outputs", "statistics", "ko-sroberta-multitask", "Pure_Metrics_Data_Final_5D.csv"))
d = m.merge(key.rename(columns={"sample_index": "pkl_index"})[["genre", "pkl_index", "wps_traj"]], on=["genre", "pkl_index"])
for met, name in MET:
    print(f"  {name:6s} rho = {d[[met, 'wps_traj']].corr(method='spearman').iloc[0, 1]:+.3f}")
print("  wps_traj median FF / PP:", d[d.is_flash == 1].wps_traj.median(), "/", d[d.is_flash == 0].wps_traj.median())
