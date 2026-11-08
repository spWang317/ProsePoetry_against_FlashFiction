# -*- coding: utf-8 -*-
"""The five measures of a trajectory and the per-encoder statistics.

Run from the repository root (one encoder per call):
    python code/measures_and_statistics.py --ff_pkl <flash fiction> --pp_pkl <prose poems> --out_dir <folder>

Reads two embedding files (data/embeddings/). Writes to --out_dir the measures of every text, the per-measure
tests (Mann-Whitney U with BH-FDR, Cliff's delta, length-controlled OLS with HC3 errors), the correlations and
variance inflation factors of the measures, the multivariate comparison (PERMANOVA, dispersion) and the ablation.
"""
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu, levene
from scipy.signal import periodogram
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.tools import add_constant
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =========================================================
# 1. Utility functions
# =========================================================
def _compute_vif_table(X_df):
    """Centered VIF calculation with constant for collinearity diagnostics"""
    X_std = StandardScaler().fit_transform(X_df.values.astype(float))
    X_with_const = add_constant(X_std)
    vifs = [{"Metric": col, "VIF": float(variance_inflation_factor(X_with_const, i))}
            for i, col in enumerate(["const"] + list(X_df.columns)) if col != "const"]
    return pd.DataFrame(vifs).sort_values("VIF", ascending=False)

def _permanova_custom(distance_matrix, groups, n_perm=4999, seed=42):
    """Implementation of Anderson (2001) PERMANOVA"""
    n, D = len(groups), distance_matrix
    uniq = np.unique(groups)
    num_groups = len(uniq)

    def calculate_ssw(labels):
        ss_w = 0.0
        for g in uniq:
            idx = np.where(labels == g)[0]
            n_g = len(idx)
            if n_g > 1:
                dist_sub = D[np.ix_(idx, idx)]
                ss_w += np.sum(dist_sub**2) / (2 * n_g)
        return ss_w

    ss_w_obs = calculate_ssw(groups)
    ss_total = np.sum(D**2) / (2 * n)
    ss_a_obs = ss_total - ss_w_obs
    df_a, df_w = num_groups - 1, n - num_groups
    f_obs = (ss_a_obs / df_a) / (ss_w_obs / df_w) if ss_w_obs != 0 else 0.0
    
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        perm_ssw = calculate_ssw(rng.permutation(groups))
        perm_f = ((ss_total - perm_ssw) / df_a) / (perm_ssw / df_w) if perm_ssw != 0 else 0.0
        if perm_f >= f_obs: ge += 1
    return f_obs, (ge + 1) / (n_perm + 1), ss_a_obs, ss_w_obs

def dispersion_check_full(X, groups, out_csv=None):
    """Multivariate homogeneity of groups (Betadisper diagnostic)"""
    groups, uniq = np.asarray(groups), np.unique(groups)
    centroids = {g: X[groups == g].mean(axis=0) for g in uniq}
    dists = np.array([np.linalg.norm(X[i] - centroids[groups[i]]) for i in range(len(groups))])
    
    rows = []
    dg_list = []
    for g in uniq:
        dg = dists[groups == g]
        dg_list.append(dg)
        rows.append({"genre": g, "n": len(dg), "mean_dist": np.mean(dg), 
                     "median_dist": np.median(dg), "std_dist": np.std(dg)})
    
    lev_stat, lev_p = levene(*dg_list, center="median")
    out_df = pd.DataFrame(rows)
    out_df["levene_stat"] = lev_stat
    out_df["levene_p"] = lev_p
    
    if out_csv: out_df.to_csv(out_csv, index=False)
    return {"levene_stat": lev_stat, "levene_p": lev_p}, out_df

# =========================================================
# 2. The five measures
# =========================================================
# A move between two identical consecutive sentences has no direction. Such moves differ from
# zero only by floating-point noise (below 2e-6 for the encoders used; every other move exceeds 0.06)
# and are left out of the curvature.
ZERO_STEP = 1e-4

def compute_5d_metrics(embs):
    T = embs.shape[0]
    if T < 5: return None
    
    # [1] Displacement
    delta = np.linalg.norm(embs[-1] - embs[0])
    
    # [2] Transition volatility
    cos_sims = np.clip([np.dot(embs[i], embs[i+1]) for i in range(T-1)], -1.0, 1.0)
    surprisals = -np.log((cos_sims + 1.0) / 2.0 + 1e-12)
    surprisal_std = np.std(surprisals)
    
    # [3] Global resonance
    ssm = 1.0 - squareform(pdist(embs, 'cosine'))
    np.fill_diagonal(ssm, 0)
    global_resonance = np.mean(ssm[np.triu_indices(T, k=1)])
    
    # [4] Curvature
    steps = embs[1:] - embs[:-1]
    step_norms = np.linalg.norm(steps, axis=1)
    angles = []
    for i in range(len(steps) - 1):
        if step_norms[i] > ZERO_STEP and step_norms[i+1] > ZERO_STEP:
            norm_p = step_norms[i] * step_norms[i+1]
            angles.append(np.arccos(np.clip(np.dot(steps[i], steps[i+1]) / norm_p, -1.0, 1.0)))
    curvature = np.mean(angles) if len(angles) >= 2 else np.nan
    
    # [5] Spectral centroid
    x = step_norms - np.mean(step_norms)
    f, psd = periodogram(x, fs=1.0, nfft=64)
    pos_mask = f > 0
    psd_total = np.sum(psd[pos_mask])
    spec_centroid = np.sum(f[pos_mask] * psd[pos_mask]) / psd_total if psd_total > 1e-12 else 0.0
        
    return {"delta": delta, "surprisal_std": surprisal_std, 
            "global_resonance": global_resonance, "curvature": curvature, 
            "spec_centroid": spec_centroid, "psd_total": psd_total}

# =========================================================
# 3. Main Analysis Execution
# =========================================================
DEFAULT_FF_PKL = "data/embeddings/FlashFiction_embs_ko-sroberta-multitask.pkl"
DEFAULT_PP_PKL = "data/embeddings/ProsePoem_embs_ko-sroberta-multitask.pkl"
DEFAULT_OUT_DIR = "outputs/statistics/ko-sroberta-multitask"

def run_analysis(ff_pkl=DEFAULT_FF_PKL, pp_pkl=DEFAULT_PP_PKL, out_dir=DEFAULT_OUT_DIR):
    """ff_pkl / pp_pkl: embedding files; out_dir: where tables are written."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Load the embedding files
    targets = [(ff_pkl, "FlashFiction"), (pp_pkl, "ProsePoem")]
    
    all_data = []
    metrics_list = ["delta", "surprisal_std", "global_resonance", "curvature", "spec_centroid"]
    
    print("[Step 1] Loading Pre-computed Embeddings and Computing 5D Metrics...")
    for f_name, genre in targets:
        pkl_path = f_name
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(pkl_path)
            
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Position of each document in the analysis sample.
        doc_ids = data["pkl_index"]
        for i in tqdm(range(len(data['embeddings'])), desc=f"Analyzing {genre}"):
            embs = data['embeddings'][i]
            if embs is None or len(embs) < 5: 
                continue
            
            # Compute geometric metrics from vectors
            m = compute_5d_metrics(embs)
            if m:
                # Length covariate for OLS: number of sentences in the embedded trajectory.
                # n_sent_matching keeps the sentence count recorded at the matching stage.
                m.update({
                    "genre": genre, 
                    "is_flash": 1 if genre == "FlashFiction" else 0,
                    "n_sent": float(embs.shape[0]), 
                    "log_n_sent": float(np.log1p(embs.shape[0])),
                    "n_sent_matching": float(data['n_sentences'][i]),
                    "pkl_index": int(doc_ids[i])
                })
                all_data.append(m)

    # Export Pure Metrics CSV
    df = pd.DataFrame(all_data).dropna(subset=metrics_list)
    df.to_csv(os.path.join(out_dir, "Pure_Metrics_Data_Final_5D.csv"), index=False)

    # Log: share of texts with zero spectral power
    sc_zero_ratio = (df['spec_centroid'] == 0).mean()
    with open(os.path.join(out_dir, "Stability_Log.txt"), "w") as f:
        f.write(f"Total Samples: {len(df)}\n")
        f.write(f"SpecCentroid Zero Ratio: {sc_zero_ratio:.4f}\n")
        f.write(f"PSD Total Mean: {df['psd_total'].mean():.6f}\n")

    print("\n[Step 2] Per-measure tests and regression...")
    summary_rows = []
    p_vals = []
    for met in metrics_list:
        f_vals = df[df["genre"]=="FlashFiction"][met].values
        p_vals_g = df[df["genre"]=="ProsePoem"][met].values
        
        # Mann-Whitney U Test
        u, p = mannwhitneyu(f_vals, p_vals_g, alternative="two-sided")
        p_vals.append(p)
        
        # OLS Regression controlling for length effect
        reg = smf.ols(f"{met} ~ is_flash + log_n_sent", data=df).fit(cov_type="HC3")
        ci = reg.conf_int().loc["is_flash"]
        
        summary_rows.append({
            "Metric": met, 
            "FF_Med": float(np.median(f_vals)), 
            "PP_Med": float(np.median(p_vals_g)),
            "MWU_U": float(u), 
            "MWU_p": float(p), 
            "Cliffs_Delta": (2.0*u/(len(f_vals)*len(p_vals_g)))-1.0,
            "OLS_Genre_Coef_Log": float(reg.params["is_flash"]),
            "OLS_Genre_CI_Lower_Log": float(ci[0]), 
            "OLS_Genre_CI_Upper_Log": float(ci[1]),
            "OLS_Genre_p_Log": float(reg.pvalues["is_flash"]), 
            "R2_Log": float(reg.rsquared)
        })
    
    # Apply FDR correction (Multiple testing)
    fdr_p = multipletests(p_vals, method="fdr_bh")[1]
    for i, r in enumerate(summary_rows): r["MWU_p_FDR"] = float(fdr_p[i])
    
    # Save Final Summary Table
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "Final_Summary_Table_5D.csv"), index=False)

    print("\n[Step 3] VIF, PERMANOVA, and dispersion...")
    # Relations between the measures
    _compute_vif_table(df[metrics_list]).to_csv(os.path.join(out_dir, "Metric_VIF_5D.csv"), index=False)
    df[metrics_list].corr(method='spearman').to_csv(os.path.join(out_dir, "Metric_Orthogonality_Spearman.csv"))
    
    # Multivariate Testing
    X_all = StandardScaler().fit_transform(df[metrics_list].values)
    disp_res, _ = dispersion_check_full(X_all, df["genre"].values, out_csv=os.path.join(out_dir, "Dispersion_Check_5D.csv"))
    f_stat, p_perm, ssa, ssw = _permanova_custom(squareform(pdist(X_all)), df["genre"].values)
    
    with open(os.path.join(out_dir, "PERMANOVA_Result_5D.txt"), "w") as f:
        f.write(f"pseudo-F: {f_stat:.6f}\np-value: {p_perm:.6f}\nSS_between: {ssa:.6f}\nSS_within: {ssw:.6f}\n\n")
        f.write(f"Levene_stat: {disp_res['levene_stat']:.6f}\nLevene_p: {disp_res['levene_p']:.6f}")

    print("\n[Step 4] Ablation (random forest)...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"))])
    X_raw, y = df[metrics_list].values, df["is_flash"].values
    
    ab_rows = []
    # Baseline: Full Model
    f1s_f, aucs_f = [], []
    for tr, te in skf.split(X_raw, y):
        p = clone(pipe).fit(X_raw[tr], y[tr])
        f1s_f.append(f1_score(y[te], p.predict(X_raw[te])))
        aucs_f.append(roc_auc_score(y[te], p.predict_proba(X_raw[te])[:, 1]))
    
    full_f1, full_auc = np.mean(f1s_f), np.mean(aucs_f)
    ab_rows.append({"Dropped": "None (Full)", "F1_mean": full_f1, "F1_std": np.std(f1s_f), 
                    "AUROC_mean": full_auc, "AUROC_std": np.std(aucs_f)})

    # Leave-One-Out Ablation
    for i, met in enumerate(metrics_list):
        X_sub = np.delete(X_raw, i, axis=1)
        f1s_s, aucs_s = [], []
        for tr, te in skf.split(X_sub, y):
            p = clone(pipe).fit(X_sub[tr], y[tr])
            f1s_s.append(f1_score(y[te], p.predict(X_sub[te])))
            aucs_s.append(roc_auc_score(y[te], p.predict_proba(X_sub[te])[:, 1]))
        
        m_f1, m_auc = np.mean(f1s_s), np.mean(aucs_s)
        ab_rows.append({
            "Dropped": met, "F1_mean": m_f1, "F1_std": np.std(f1s_s), 
            "AUROC_mean": m_auc, "AUROC_std": np.std(aucs_s),
            "Drop_F1": full_f1 - m_f1, "Drop_AUROC": full_auc - m_auc
        })
    
    pd.DataFrame(ab_rows).to_csv(os.path.join(out_dir, "Ablation_Study_Results_5D.csv"), index=False)
    print(f"\nAll analysis stages complete. Results saved in {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="5D metrics and statistics from sentence-embedding files.")
    ap.add_argument("--ff_pkl", default=DEFAULT_FF_PKL)
    ap.add_argument("--pp_pkl", default=DEFAULT_PP_PKL)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    if (args.ff_pkl != DEFAULT_FF_PKL or args.pp_pkl != DEFAULT_PP_PKL) and \
            os.path.abspath(args.out_dir) == os.path.abspath(DEFAULT_OUT_DIR):
        ap.error("--out_dir must differ from the default when other embedding files are analysed")
    run_analysis(args.ff_pkl, args.pp_pkl, args.out_dir)
