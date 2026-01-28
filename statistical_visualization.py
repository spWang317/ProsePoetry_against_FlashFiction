# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# ==========================================
# 1. Environment and Font Configuration
# ==========================================
plt.switch_backend("agg")
sns.set_context("paper", font_scale=1.4)
sns.set_style("white")
plt.rcParams['axes.unicode_minus'] = False 

# Directory Configuration
DATA_DIR = "./Final_Research_Output_5D_Restored"
FIGURE_DIR = "./Figures_Final_5D_Hardened"
os.makedirs(FIGURE_DIR, exist_ok=True)

# Metric Label Standardization (Line-break versions included for plotting)
CORE_METRICS = ["delta", "surprisal_std", "global_resonance", "curvature", "spec_centroid"]
METRIC_LABELS = ["Displacement", "Volatility", "Resonance", "Curvature", "Spectral Centroid"]
METRIC_LABELS_BR = [L.replace(" ", "\n") for L in METRIC_LABELS] # Apply line breaks
LABEL_MAP = dict(zip(CORE_METRICS, METRIC_LABELS_BR))

GENRE_COLORS = {"FlashFiction": "#E63946", "ProsePoem": "#457B9D"}
GENRE_ORDER = ["FlashFiction", "ProsePoem"]

def clean_df(df):
    """Convert genre labels from numeric to string formats"""
    if 'genre' in df.columns and df['genre'].dtype in [np.int64, np.int32]:
        mapping = {1: "FlashFiction", 0: "ProsePoem"}
        df['genre'] = df['genre'].map(mapping)
    return df

# =========================================================
# Fig 1. Structure Validation: Orthogonality and VIF
# =========================================================
def draw_fig1_ortho(corr_csv, vif_csv, out_path):
    if not (os.path.exists(corr_csv) and os.path.exists(vif_csv)): return
    corr_df = pd.read_csv(corr_csv, index_col=0).reindex(index=CORE_METRICS, columns=CORE_METRICS)
    vif_df = pd.read_csv(vif_csv).set_index('Metric').reindex(CORE_METRICS).reset_index()
    
    # Apply multi-line labels
    corr_df.index = [LABEL_MAP[i] for i in corr_df.index]
    corr_df.columns = [LABEL_MAP[c] for c in corr_df.columns]
    vif_df['Metric'] = vif_df['Metric'].map(LABEL_MAP)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={'width_ratios': [1.2, 1]})
    
    sns.heatmap(corr_df, annot=True, cmap="RdBu_r", center=0, ax=ax1, fmt=".2f", cbar_kws={'shrink': 0.8})
    # Set title padding to 20
    ax1.set_title("A. 5D Feature Orthogonality", fontweight='bold', fontsize=26, pad=20)
    
    sns.barplot(data=vif_df, x='VIF', y='Metric', ax=ax2, palette="Greys_r")
    ax2.axvline(5, color='red', linestyle='--', linewidth=2)
    ax2.set_title("B. Redundancy Check (VIF)", fontweight='bold', fontsize=26, pad=20)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()

# =========================================================
# Fig 2. Multivariate Dispersion Analysis
# =========================================================
def draw_fig2_dispersion(disp_csv, out_path):
    if not os.path.exists(disp_csv): return
    d = clean_df(pd.read_csv(disp_csv))
    d = d.set_index("genre").reindex(GENRE_ORDER).reset_index()

    plt.figure(figsize=(7, 6))
    sns.barplot(data=d, x="genre", y="mean_dist", palette=GENRE_COLORS, order=GENRE_ORDER)
    plt.errorbar(x=np.arange(len(d)), y=d["mean_dist"], yerr=d["std_dist"], 
                 fmt='none', c='black', capsize=8, elinewidth=2)

    p = d["levene_p"].iloc[0]
    plt.title(f"Multivariate Dispersion (Levene p={p:.3f})", fontsize=18, fontweight='bold', pad=20)
    plt.ylabel("Distance to Centroid")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()

# =========================================================
# Fig 3. Distributional Comparison across Metrics
# =========================================================
def draw_fig3_distribution(data_csv, out_path):
    if not os.path.exists(data_csv): return
    df = clean_df(pd.read_csv(data_csv))
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 6)) 
    for i, m in enumerate(CORE_METRICS):
        sns.violinplot(data=df, x="genre", y=m, palette=GENRE_COLORS, ax=axes[i], inner="quartile", order=GENRE_ORDER)
        axes[i].set_title(METRIC_LABELS[i], fontsize=18, fontweight='bold', pad=15)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()

# =========================================================
# Fig 4. Genre Effects Controlling for Length
# =========================================================
def draw_fig4_coefficients(summary_csv, out_path):
    if not os.path.exists(summary_csv): return
    df = pd.read_csv(summary_csv)
    df["Metric"] = pd.Categorical(df["Metric"], categories=CORE_METRICS, ordered=True)
    df_sorted = df.sort_values("Metric", ascending=False).reset_index(drop=True)
    
    y_pos = np.arange(len(df_sorted))
    x = df_sorted["OLS_Genre_Coef_Log"].values
    lo, hi = df_sorted["OLS_Genre_CI_Lower_Log"].values, df_sorted["OLS_Genre_CI_Upper_Log"].values

    plt.figure(figsize=(11, 8))
    plt.errorbar(x, y_pos, xerr=[x - lo, hi - x], fmt='o', color='black', ecolor='gray', capsize=8, markersize=12)
    plt.axvline(0, linestyle='--', color='red', alpha=0.6)
    
    plt.yticks(y_pos, [LABEL_MAP[m] for m in df_sorted["Metric"]], fontsize=16)
    
    # Allocate upper margin for significance markers
    plt.ylim(-0.5, len(df_sorted) + 0.5)
    plt.title("Genre Effect Controlling for Length", fontsize=22, pad=35)
    
    for i, p in enumerate(df_sorted["OLS_Genre_p_Log"]):
        if p < 0.05:
            stars = "*" if p > 0.01 else "**" if p > 0.001 else "***"
            plt.text(x[i], i + 0.25, stars, ha='center', va='bottom', fontsize=18, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()

# =========================================================
# Fig 5. Standardized Radar Profiles
# =========================================================
def draw_fig5_radar(data_csv, out_path):
    if not os.path.exists(data_csv): return
    df = clean_df(pd.read_csv(data_csv))
    radar_df = df.groupby("genre")[CORE_METRICS].median()
    radar_norm = (radar_df - df[CORE_METRICS].mean()) / df[CORE_METRICS].std().replace(0, 1e-12)
    
    angles = np.linspace(0, 2*np.pi, len(CORE_METRICS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for genre in GENRE_ORDER:
        if genre not in radar_norm.index: continue
        values = radar_norm.loc[genre].tolist() + [radar_norm.loc[genre].iloc[0]]
        ax.plot(angles, values, color=GENRE_COLORS[genre], linewidth=4, label=genre, marker='o')
        ax.fill(angles, values, color=GENRE_COLORS[genre], alpha=0.18)
    
    # Initialize basic labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRIC_LABELS_BR, fontsize=15)
    
    # Fine-tune individual label offsets
    labels = ax.get_xticklabels()
    for label in labels:
        text = label.get_text().replace("\n", " ")
        if "Displacement" in text:
            label.set_va('center')
            label.set_position((0, 0.12)) # Offset outward
        elif "Volatility" in text:
            label.set_position((0, -0.05)) # Offset inward
        else:
            label.set_position((0, 0.05)) # Default spacing

    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=14)
    plt.title("Standardized 5D Dynamics Profiles", fontsize=22, pad=40)
    plt.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close()

# =========================================================
# Fig 6. Phase Space: All Pairwise Combinations (10 total)
# =========================================================
def draw_fig6_phase_space(data_csv, out_path_base):
    if not os.path.exists(data_csv): return
    df = clean_df(pd.read_csv(data_csv))
    
    # Generate all pairwise combinations from CORE_METRICS (5C2 = 10)
    phase_pairs = list(itertools.combinations(CORE_METRICS, 2))

    print(f"📊 Generating {len(phase_pairs)} phase space plots...")

    for i, (xx, yy) in enumerate(phase_pairs, 1):
        plt.figure(figsize=(8, 7))
        
        # Visualize scatter plots and Kernel Density Estimations (KDE)
        sns.scatterplot(data=df, x=xx, y=yy, hue="genre", palette=GENRE_COLORS, alpha=0.15, legend=True)
        sns.kdeplot(data=df, x=xx, y=yy, hue="genre", palette=GENRE_COLORS, levels=5, alpha=0.6)
        
        # Configure labels and titles using LABEL_MAP
        x_label = LABEL_MAP[xx].replace("\n", " ")
        y_label = LABEL_MAP[yy].replace("\n", " ")
        
        plt.title(f"{x_label} vs {y_label}", fontweight='bold', fontsize=18, pad=15)
        plt.xlabel(x_label, fontsize=15)
        plt.ylabel(y_label, fontsize=15)
        plt.legend(frameon=True, loc='upper right', fontsize=12)
        
        plt.tight_layout()
        
        # Generate filename: index_metric1_vs_metric2
        filename = f"Phase_{i:02d}_{xx}_vs_{yy}"
        save_path = out_path_base.replace(".png", f"_{filename}.png")
        
        plt.savefig(save_path, dpi=300)
        plt.close()

# =========================================================
# Fig 7. Diagnostic Ablation Study
# =========================================================
def draw_fig7_ablation(ablation_csv, out_path):
    if not os.path.exists(ablation_csv): return
    df = pd.read_csv(ablation_csv)
    order = ["None (Full)"] + CORE_METRICS
    df["Dropped"] = pd.Categorical(df["Dropped"], categories=order, ordered=True)
    df = df.sort_values("Dropped")

    # Metric name mapping (Retaining 'None (Full)')
    abl_label_map = {**LABEL_MAP, "None (Full)": "None (Full)"}

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    for j, (met, tit, col) in enumerate([("F1_mean", "A. F1 Score", "viridis"), ("AUROC_mean", "B. AUROC", "magma")]):
        sns.barplot(data=df, x="Dropped", y=met, ax=axes[0, j], palette=col)
        axes[0, j].set_title(tit, fontweight="bold", fontsize=18)
        axes[0, j].set_ylim(0, 1.05)
        axes[0, j].axhline(df[df["Dropped"]=="None (Full)"][met].values[0], color='red', linestyle='--', alpha=0.5)

    for j, (met, tit, col) in enumerate([("Drop_F1", "C. Drop in F1", "#1D3557"), ("Drop_AUROC", "D. Drop in AUROC", "#E63946")]):
        sns.barplot(data=df[df["Dropped"] != "None (Full)"], x="Dropped", y=met, ax=axes[1, j], color=col)
        axes[1, j].set_title(tit, fontweight="bold", fontsize=18)

    for ax in axes.flatten(): 
        ax.set_xticklabels([abl_label_map.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()], rotation=30, fontsize=12)
        ax.set_xlabel("")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()

# =========================================================
# Main Execution
# =========================================================
if __name__ == "__main__":
    print(f"🚀 [Visualizing] Processing final 5D hardened figures...")
    raw_path = os.path.join(DATA_DIR, "Pure_Metrics_Data_Final_5D.csv")
    summary_path = os.path.join(DATA_DIR, "Final_Summary_Table_5D.csv")
    ablation_path = os.path.join(DATA_DIR, "Ablation_Study_Results_5D.csv")
    corr_path = os.path.join(DATA_DIR, "Metric_Orthogonality_Spearman.csv")
    vif_path = os.path.join(DATA_DIR, "Metric_VIF_5D.csv")
    disp_path = os.path.join(DATA_DIR, "Dispersion_Check_5D.csv")

    # Execute visualization pipeline
    draw_fig1_ortho(corr_path, vif_path, os.path.join(FIGURE_DIR, "Fig1_Structure_Validation.png"))
    draw_fig2_dispersion(disp_path, os.path.join(FIGURE_DIR, "Fig2_Dispersion_Check.png"))
    draw_fig3_distribution(raw_path, os.path.join(FIGURE_DIR, "Fig3_Distributions_5D.png"))
    draw_fig4_coefficients(summary_path, os.path.join(FIGURE_DIR, "Fig4_OLS_Coefficients.png"))
    draw_fig5_radar(raw_path, os.path.join(FIGURE_DIR, "Fig5_Radar_Profile.png"))
    
    # Fig 6: Generate modularized individual panels
    draw_fig6_phase_space(raw_path, os.path.join(FIGURE_DIR, "Fig6_Phase.png"))
    
    draw_fig7_ablation(ablation_path, os.path.join(FIGURE_DIR, "Fig7_Ablation_Necessity.png"))
    
    print(f"✅ [Success] All hardened figures saved in: {FIGURE_DIR}")