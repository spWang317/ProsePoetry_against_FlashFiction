# -*- coding: utf-8 -*-
"""Figures of the manuscript and the Supplementary Information.

Run from the repository root:
    python code/06_figures.py --out_dir outputs/figures

Inputs: outputs/tables/effect_sizes.csv, outputs/statistics/<encoder>/Pure_Metrics_Data_Final_5D.csv,
        data/corpus_table.csv (sentence counts of every text), data/sample_rows.csv.
Outputs (PDF and PNG): Figure1 (schematic of the five measures), Figure2 (length distributions),
        Figure3 (distributions of the measures), Figure4 (effect sizes under three encoders),
        FigureS1 (distance to the genre centroid), FigureS2 (pairwise relations between the measures).
"""
import os, argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from scipy.stats import gaussian_kde
from scipy.signal import periodogram
from sklearn.preprocessing import StandardScaler

FF, PP = "#C0392B", "#2E6FB7"            # flash fiction, prose poetry
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"
GENRES = [("FlashFiction", "Flash fiction", FF), ("ProsePoem", "Prose poetry", PP)]
METRICS = [("delta", r"Displacement $\delta$"), ("surprisal_std", r"Transition volatility $\sigma_S$"),
           ("global_resonance", r"Global resonance $\mathcal{R}$"), ("curvature", r"Curvature $\kappa$"),
           ("spec_centroid", r"Spectral centroid $\Omega$")]
ENCODERS = [("ko-sroberta-multitask", "ko-sroberta-multitask", "o"), ("bge-m3", "BGE-M3", "^"),
            ("multilingual-e5-large", "multilingual-e5-large", "s")]
NUM, CORE = "outputs/tables/", "outputs/statistics/"

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})


def save(fig, out_dir, name):
    fig.savefig(os.path.join(out_dir, name + ".pdf"), bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(os.path.join(out_dir, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def letter(ax, s, x=-0.02, y=1.08):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right", color=INK)


# ------------------------------------------------------------------ Figure 1: schematic
def path_plot(ax, pts, dashed_ends=False, all_pairs=False):
    pts = np.asarray(pts)
    if all_pairs:
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                ax.plot(*zip(pts[i], pts[j]), color=GRID, lw=0.5, zorder=1)
    for a, b in zip(pts[:-1], pts[1:]):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, shrinkA=2.5, shrinkB=2.5, mutation_scale=7), zorder=3)
    ax.scatter(pts[:, 0], pts[:, 1], s=14, color=INK, zorder=4)
    if dashed_ends:
        ax.plot(*zip(pts[0], pts[-1]), color=FF, lw=1.1, ls=(0, (3, 2)), zorder=2)
        ax.text(pts[0][0] - 0.05, pts[0][1] - 0.05, r"$e_1$", fontsize=7, ha="right", va="top", color=INK)
        ax.text(pts[-1][0] + 0.05, pts[-1][1] + 0.05, r"$e_T$", fontsize=7, ha="left", va="bottom", color=INK)
    ax.set_xlim(-0.15, 1.15); ax.set_ylim(-0.15, 1.15); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(GRID)


def figure1(out_dir):
    fig, axes = plt.subplots(2, 5, figsize=(7.4, 3.5), gridspec_kw={"wspace": 0.3, "hspace": 0.34})
    titles = ["Displacement\n" + r"$\delta$", "Transition volatility\n" + r"$\sigma_S$", "Global resonance\n" + r"$\mathcal{R}$",
              "Curvature\n" + r"$\kappa$", "Spectral centroid\n" + r"$\Omega$"]
    # a displacement
    path_plot(axes[0, 0], [(0.2, 0.3), (0.45, 0.8), (0.85, 0.6), (0.75, 0.2), (0.45, 0.12), (0.33, 0.42)], dashed_ends=True)
    path_plot(axes[1, 0], [(0.1, 0.15), (0.3, 0.45), (0.25, 0.75), (0.55, 0.6), (0.75, 0.85), (0.9, 0.7)], dashed_ends=True)
    # b transition volatility: sizes of the change between neighbouring sentences
    for ax, s in [(axes[0, 1], [0.5, 0.55, 0.48, 0.52, 0.5, 0.54, 0.47]), (axes[1, 1], [0.15, 0.2, 0.95, 0.18, 0.7, 0.12, 0.85])]:
        ax.bar(range(1, 8), s, width=0.62, color=MUTED)
        ax.set_ylim(0, 1.05); ax.set_xticks([1, 4, 7]); ax.set_yticks([])
        ax.set_xlabel("sentence step $t$", labelpad=1); ax.set_ylabel("$s_t$", labelpad=1)
    # c global resonance
    rng = np.random.default_rng(3)
    path_plot(axes[0, 2], np.clip(rng.uniform(0.02, 0.98, (7, 2)), 0, 1), all_pairs=True)
    path_plot(axes[1, 2], 0.5 + 0.11 * rng.standard_normal((7, 2)), all_pairs=True)
    # d curvature: angle between two successive moves
    for ax, third in [(axes[0, 3], (0.82, 0.95)), (axes[1, 3], (0.38, 0.22))]:
        p1, p2, p3 = np.array((0.1, 0.25)), np.array((0.62, 0.52)), np.array(third)
        path_plot(ax, [p1, p2, p3])
        ext = p2 + 0.42 * (p2 - p1) / np.linalg.norm(p2 - p1)
        ax.plot(*zip(p2, ext), color=MUTED, lw=0.8, ls=(0, (2, 2)))
        a0 = np.degrees(np.arctan2(*(ext - p2)[::-1])); a1 = np.degrees(np.arctan2(*(p3 - p2)[::-1]))
        lo, hi = sorted([a0, a1])
        if hi - lo > 180:
            lo, hi = hi, lo + 360
        ax.add_patch(Arc(p2, 0.34, 0.34, theta1=lo, theta2=hi, color=FF, lw=1.1))
        mid = np.radians((lo + hi) / 2)
        ax.text(p2[0] + 0.27 * np.cos(mid), p2[1] + 0.27 * np.sin(mid), r"$\theta_t$", fontsize=7, ha="center", va="center", color=INK)
    # e spectral centroid: step sizes and their periodogram
    t = np.arange(24)
    for ax, a in [(axes[0, 4], 0.5 + 0.35 * np.sin(2 * np.pi * t / 12)), (axes[1, 4], 0.5 + 0.35 * np.where(t % 2 == 0, 1, -1))]:
        ax.plot(t + 1, a, color=INK, lw=1.0, marker="o", ms=2.2)
        ax.set_ylim(0, 2.1); ax.set_xticks([1, 12, 24]); ax.set_yticks([])
        ax.set_xlabel("sentence step $t$", labelpad=1); ax.set_ylabel("$a_t$", labelpad=1)
        f, p = periodogram(a - a.mean(), fs=1.0, nfft=64)
        ins = ax.inset_axes([0.42, 0.62, 0.56, 0.34])
        ins.fill_between(f[1:], p[1:] / p[1:].max(), color=MUTED, lw=0)
        c = (f[1:] * p[1:]).sum() / p[1:].sum()
        ins.axvline(c, color=FF, lw=1.1)
        ins.set_xlim(0, 0.5); ins.set_ylim(0, 1.15); ins.set_xticks([0, 0.5]); ins.set_yticks([])
        ins.tick_params(labelsize=5.5, length=2, pad=1)
        ins.text(0.25, 1.22, "spectrum of $a_t$", fontsize=5.5, ha="center", va="bottom", color=MUTED)
    for k in range(5):
        axes[0, k].set_title(titles[k], fontsize=8, pad=8, linespacing=1.25)
        letter(axes[0, k], "abcde"[k], x=-0.02, y=1.17)
    for r, lab in enumerate(["lower value", "higher value"]):
        axes[r, 0].text(-0.22, 0.5, lab, transform=axes[r, 0].transAxes, rotation=90, va="center", ha="center", fontsize=8, color=INK)
    save(fig, out_dir, "Figure1")


# ------------------------------------------------------------------ Figure 2: lengths
def density(ax, values, color, log=False, grid=None):
    v = np.asarray(values, float)
    x = np.log10(v) if log else v
    k = gaussian_kde(x)
    g = np.linspace(x.min(), x.max(), 300) if grid is None else grid
    y = k(g)
    ax.fill_between(10 ** g if log else g, y, color=color, alpha=0.28, lw=0)
    ax.plot(10 ** g if log else g, y, color=color, lw=1.1)
    ax.axvline(np.median(v), color=color, lw=0.9, ls=(0, (3, 2)))


def figure2(out_dir):
    cnt = pd.read_csv("data/corpus_table.csv", low_memory=False)
    pool = cnt[(cnt.n_sent_source >= 5) & (cnt.n_sent_traj >= 5)]
    sample = pd.read_csv("data/sample_rows.csv")[["genre", "row"]].merge(cnt, on=["genre", "row"])
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 3.9), gridspec_kw={"wspace": 0.3, "hspace": 0.55})
    cols = [(pool, "n_sent_source", "wps_source", "Texts eligible for matching", "Sentences (source text)", "Words per sentence (source text)", True),
            (sample, "n_sent_source", "wps_source", "Sample: matching variables", "Sentences (source text)", "Words per sentence (source text)", False),
            (sample, "n_sent_traj", "wps_traj", "Sample: trajectories", "Sentences (trajectory)", "Words per sentence (trajectory)", False)]
    for c, (d, ns, ws, title, xl1, xl2, log) in enumerate(cols):
        for g, lab, col in GENRES:
            v = d[d.genre == g]
            density(axes[0, c], v[ns], col, log=log)
            density(axes[1, c], v[ws].clip(upper=40), col)
        if log:
            axes[0, c].set_xscale("log")
        axes[0, c].set_xlabel(xl1); axes[1, c].set_xlabel(xl2)
        axes[0, c].set_title(title, fontsize=8, pad=8)
        letter(axes[0, c], "abc"[c], x=-0.04, y=1.05)
        for r in (0, 1):
            axes[r, c].set_yticks([]); axes[r, c].set_ylim(bottom=0)
            axes[r, c].set_ylabel("Density" if c == 0 else "")
    for r, key in ((0, "n"), (1, "w")):       # the two analysis-sample columns share their x range
        lo = min(axes[r, 1].get_xlim()[0], axes[r, 2].get_xlim()[0]); hi = max(axes[r, 1].get_xlim()[1], axes[r, 2].get_xlim()[1])
        axes[r, 1].set_xlim(lo, hi); axes[r, 2].set_xlim(lo, hi)
    handles = [plt.Line2D([], [], color=col, lw=1.4, label=lab) for _, lab, col in GENRES]
    handles.append(plt.Line2D([], [], color=MUTED, lw=0.9, ls=(0, (3, 2)), label="median"))
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), fontsize=8)
    save(fig, out_dir, "Figure2")


# ------------------------------------------------------------------ Figure 3: distributions of the measures
def figure3(out_dir):
    d = pd.read_csv(CORE + "ko-sroberta-multitask/Pure_Metrics_Data_Final_5D.csv")
    fig, axes = plt.subplots(1, 5, figsize=(7.4, 2.7), gridspec_kw={"wspace": 0.7})
    for k, (m, lab) in enumerate(METRICS):
        ax = axes[k]
        data = [d.loc[d.genre == g, m].values for g, _, _ in GENRES]
        parts = ax.violinplot(data, positions=[0, 1], widths=0.82, showextrema=False)
        for body, (_, _, col) in zip(parts["bodies"], GENRES):
            body.set_facecolor(col); body.set_alpha(0.3); body.set_edgecolor(col); body.set_linewidth(0.8)
        for x, v, (_, _, col) in zip([0, 1], data, GENRES):
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([x, x], [q1, q3], color=col, lw=3.2, solid_capstyle="butt")
            ax.plot([x - 0.17, x + 0.17], [med, med], color=INK, lw=1.2)
        if m == "curvature":
            ax.axhline(2 * np.pi / 3, color=MUTED, lw=0.8, ls=(0, (2, 2)))
            ax.text(1.66, 2 * np.pi / 3, r"$2\pi/3$", fontsize=6.5, va="center", ha="left", color=MUTED, clip_on=False)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["FF", "PP"]); ax.set_xlim(-0.6, 1.6)
        ax.set_title(lab.replace(" $", "\n$"), fontsize=8, pad=6, linespacing=1.25)
        letter(ax, "abcde"[k], x=-0.2, y=1.1)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=col, alpha=0.45, ec=col, label=f"{lab} ({short})") for (_, lab, col), short in zip(GENRES, ["FF", "PP"])]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.09), fontsize=8)
    save(fig, out_dir, "Figure3")


# ------------------------------------------------------------------ Figure 4: effect sizes under three encoders
def figure4(out_dir):
    e = pd.read_csv(NUM + "effect_sizes.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    offsets = [0.24, 0.0, -0.24]
    for k, (m, lab) in enumerate(METRICS):
        y0 = len(METRICS) - 1 - k
        if k % 2 == 0:
            ax.axhspan(y0 - 0.5, y0 + 0.5, color="#f4f4f2", lw=0, zorder=0)
        for (enc, enc_lab, marker), off in zip(ENCODERS, offsets):
            r = e[(e.encoder == enc) & (e.metric == m)].iloc[0]
            ax.plot([r.delta_CI_low, r.delta_CI_high], [y0 + off] * 2, color=INK, lw=1.0, zorder=2)
            ax.plot(r.cliffs_delta, y0 + off, marker=marker, ms=5.5, mec=INK, mew=1.0,
                    mfc=INK if r.OLS_p < 0.05 else "white", zorder=3)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(METRICS))); ax.set_yticklabels([lab for _, lab in METRICS][::-1])
    ax.set_ylim(-0.5, len(METRICS) - 0.5); ax.set_xlim(-0.32, 0.32)
    ax.set_xlabel(r"Cliff's $\Delta$")
    ax.text(0.0, -0.105, r"$\leftarrow$ larger in prose poetry", transform=ax.transAxes, fontsize=7.5, color=MUTED, ha="left", va="top")
    ax.text(1.0, -0.105, r"larger in flash fiction $\rightarrow$", transform=ax.transAxes, fontsize=7.5, color=MUTED, ha="right", va="top")
    ax.tick_params(axis="y", length=0)
    top = ax.secondary_xaxis("top", functions=(lambda x: (x + 1) / 2, lambda p: 2 * p - 1))
    top.set_xlabel("Probability that a flash-fiction text has the larger value", labelpad=5)
    top.set_xticks([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
    handles = [plt.Line2D([], [], marker=mk, ls="", ms=5.5, mec=INK, mfc=INK, label=lab) for _, lab, mk in ENCODERS]
    handles += [plt.Line2D([], [], marker="o", ls="", ms=5.5, mec=INK, mfc=INK, label=r"length-controlled $p < 0.05$"),
                plt.Line2D([], [], marker="o", ls="", ms=5.5, mec=INK, mfc="white", label=r"length-controlled $p \geq 0.05$")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7.5, handletextpad=0.4, labelspacing=0.7)
    save(fig, out_dir, "Figure4")


# ------------------------------------------------------------------ Supplementary figures
def standardised(d):
    return StandardScaler().fit_transform(d[[m for m, _ in METRICS]].values)


def figure_s1(out_dir):
    d = pd.read_csv(CORE + "ko-sroberta-multitask/Pure_Metrics_Data_Final_5D.csv")
    X = standardised(d)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    dist = {g: np.linalg.norm(X[(d.genre == g).values] - X[(d.genre == g).values].mean(axis=0), axis=1) for g, _, _ in GENRES}
    grid = np.linspace(0, max(v.max() for v in dist.values()), 300)
    for g, lab, col in GENRES:
        k = gaussian_kde(dist[g])
        ax.fill_between(grid, k(grid), color=col, alpha=0.28, lw=0); ax.plot(grid, k(grid), color=col, lw=1.1, label=lab)
        ax.axvline(np.median(dist[g]), color=col, lw=0.9, ls=(0, (3, 2)))
    ax.set_xlabel("Distance to the genre centroid (standardised measures)"); ax.set_ylabel("Density")
    ax.set_yticks([]); ax.set_ylim(bottom=0); ax.legend(fontsize=7.5)
    save(fig, out_dir, "FigureS1")


def figure_s2(out_dir):
    d = pd.read_csv(CORE + "ko-sroberta-multitask/Pure_Metrics_Data_Final_5D.csv")
    short = [r"$\delta$", r"$\sigma_S$", r"$\mathcal{R}$", r"$\kappa$", r"$\Omega$"]
    n = len(METRICS)
    fig, axes = plt.subplots(n, n, figsize=(7.2, 7.2), gridspec_kw={"wspace": 0.12, "hspace": 0.12})
    for i, (mi, _) in enumerate(METRICS):
        for j, (mj, _) in enumerate(METRICS):
            ax = axes[i, j]
            if j > i:
                ax.axis("off"); continue
            for g, lab, col in GENRES:
                v = d[d.genre == g]
                if i == j:
                    k = gaussian_kde(v[mi]); grid = np.linspace(d[mi].min(), d[mi].max(), 200)
                    ax.fill_between(grid, k(grid), color=col, alpha=0.28, lw=0); ax.plot(grid, k(grid), color=col, lw=1.0)
                else:
                    ax.scatter(v[mj], v[mi], s=3, color=col, alpha=0.22, lw=0, rasterized=True)
                    k = gaussian_kde(np.vstack([v[mj], v[mi]]))
                    gx, gy = np.meshgrid(np.linspace(d[mj].min(), d[mj].max(), 60), np.linspace(d[mi].min(), d[mi].max(), 60))
                    z = k(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
                    ax.contour(gx, gy, z, levels=np.quantile(z, [0.8, 0.93]), colors=[col], linewidths=0.9)
            ax.tick_params(labelsize=6, length=2)
            if i < n - 1: ax.set_xticklabels([])
            else: ax.set_xlabel(short[j], fontsize=9)
            if j > 0 or i == 0: ax.set_yticklabels([])
            if j == 0 and i > 0: ax.set_ylabel(short[i], fontsize=9)
            if i == j: ax.set_yticks([])
    handles = [plt.Line2D([], [], color=col, lw=1.6, label=lab) for _, lab, col in GENRES]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.9, 0.86), fontsize=8)
    save(fig, out_dir, "FigureS2")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for f in (figure1, figure2, figure3, figure4, figure_s1, figure_s2):
        f(args.out_dir)
        print("written:", f.__name__)
