# -*- coding: utf-8 -*-
"""Tables reported in the article, computed from the sentence-embedding files of three encoders.

Run from the repository root:
    python code/03_tables.py --out_dir outputs/tables

Outputs (CSV, one file per table) and a run record (run_info.json):
    length_parity.csv            sentence counts and words per sentence of the two genres
    effect_sizes.csv             Cliff's delta, probability of superiority, bootstrap CI, MWU, OLS
    permanova.csv                multivariate location and dispersion, variance explained
    classification.csv           document-level classification from length, from the words and from the five measures
    spectral_window.csv          documents whose series of move lengths exceeds the 64-point window
    metric_relations.csv         rank correlations between the five measures
"""
import os, sys, json, pickle, hashlib, argparse, warnings, platform
import numpy as np, pandas as pd
import scipy, sklearn, statsmodels
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.multitest import multipletests
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measures_and_statistics import compute_5d_metrics, _permanova_custom, dispersion_check_full

warnings.filterwarnings("ignore")
SEED = 42
METRICS = ["delta", "surprisal_std", "global_resonance", "curvature", "spec_centroid"]
GENRES = ["FlashFiction", "ProsePoem"]
EMB = "data/embeddings/"
ENCODERS = {  # label -> embedding files (flash fiction, prose poetry)
    "ko-sroberta-multitask": (EMB + "FlashFiction_embs_ko-sroberta-multitask.pkl", EMB + "ProsePoem_embs_ko-sroberta-multitask.pkl"),
    "bge-m3": (EMB + "FlashFiction_embs_bge-m3.pkl", EMB + "ProsePoem_embs_bge-m3.pkl"),
    "multilingual-e5-large": (EMB + "FlashFiction_embs_me5-large.pkl", EMB + "ProsePoem_embs_me5-large.pkl"),
}
WORD_BASELINE = EMB + "word_baseline.json"   # TF-IDF classifiers (random forest, logistic regression), computed with the source texts
N_BOOT = 10000


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_encoder(files):
    """One row per analysed document: the five measures and its lengths."""
    rows = []
    for path, genre in zip(files, GENRES):
        d = pickle.load(open(path, "rb"))
        ids = d["pkl_index"]
        for k, e in enumerate(d["embeddings"]):
            if e is None or len(e) < 5:
                continue
            e = np.asarray(e)
            m = compute_5d_metrics(e)
            m.update(genre=genre, is_flash=int(genre == "FlashFiction"), pkl_index=int(ids[k]),
                     T=int(e.shape[0]), log_n_sent=float(np.log1p(e.shape[0])),
                     n_match=float(d["n_sentences"][k]), wps_match=float(d["tokens_per_sent"][k]),
                     wps_traj=float(d["tokens_per_sent_traj"][k]))
            rows.append(m)
    df = pd.DataFrame(rows)
    assert df[METRICS].notna().all().all()
    return df


def cliffs_delta(a, b):
    """P(a > b) - P(a < b) for two samples."""
    b = np.sort(b)
    lo, hi = np.searchsorted(b, a, "left"), np.searchsorted(b, a, "right")
    return float((lo.sum() - (len(b) - hi).sum()) / (len(a) * len(b)))


def genre_tests(df):
    """Per measure: medians, Cliff's delta, Mann-Whitney U (BH-FDR over the five measures), length-controlled OLS (HC3)."""
    out, ps = [], []
    for m in METRICS:
        a, b = df.loc[df.is_flash == 1, m].values, df.loc[df.is_flash == 0, m].values
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        reg = smf.ols(f"{m} ~ is_flash + log_n_sent", data=df).fit(cov_type="HC3")
        ci = reg.conf_int().loc["is_flash"]
        ps.append(p)
        out.append({"metric": m, "FF_median": np.median(a), "PP_median": np.median(b),
                    "cliffs_delta": cliffs_delta(a, b), "MWU_U": u, "MWU_p": p,
                    "OLS_beta": reg.params["is_flash"], "OLS_CI_low": ci[0], "OLS_CI_high": ci[1],
                    "OLS_p": reg.pvalues["is_flash"]})
    for r, q in zip(out, multipletests(ps, method="fdr_bh")[1]):
        r["MWU_p_FDR"] = q
    return pd.DataFrame(out)


def length_parity(df):
    """Lengths at the matching stage and in the embedded trajectories."""
    d = df.assign(words_match=df.n_match * df.wps_match, words_traj=df["T"] * df.wps_traj)
    rows = []
    for col, label in [("n_match", "sentences, matching stage"), ("wps_match", "words per sentence, matching stage"),
                       ("words_match", "words, matching stage"), ("T", "sentences, embedded trajectory"),
                       ("wps_traj", "words per sentence, embedded trajectory"), ("words_traj", "words, embedded trajectory")]:
        a, b = d.loc[d.is_flash == 1, col], d.loc[d.is_flash == 0, col]
        rows.append({"quantity": label, "n_FF": len(a), "n_PP": len(b), "FF_median": a.median(), "PP_median": b.median(),
                     "FF_mean": a.mean(), "PP_mean": b.mean(), "MWU_p": mannwhitneyu(a, b).pvalue})
    rows.append({"quantity": "documents whose trajectory length differs from the matching-stage count",
                 "n_FF": int(((d.is_flash == 1) & (d["T"] != d.n_match)).sum()),
                 "n_PP": int(((d.is_flash == 0) & (d["T"] != d.n_match)).sum())})
    return pd.DataFrame(rows)


def effect_sizes(df, encoder):
    """Cliff's delta with a percentile bootstrap interval (documents resampled within genre)."""
    t = genre_tests(df)
    rng = np.random.default_rng(SEED)
    na, nb = int(df.is_flash.sum()), int((1 - df.is_flash).sum())
    ia, ib = rng.integers(0, na, (N_BOOT, na)), rng.integers(0, nb, (N_BOOT, nb))
    lo, hi = [], []
    for m in METRICS:
        a, b = df.loc[df.is_flash == 1, m].values, df.loc[df.is_flash == 0, m].values
        boot = np.array([cliffs_delta(a[i], b[j]) for i, j in zip(ia, ib)])
        q = np.percentile(boot, [2.5, 97.5])
        lo.append(q[0]); hi.append(q[1])
    t["delta_CI_low"], t["delta_CI_high"] = lo, hi
    t["prob_superiority_FF"] = (t.cliffs_delta + 1) / 2  # P(FF > PP) + 0.5 P(tie)
    t.insert(0, "encoder", encoder)
    return t


def permanova(df, encoder):
    X = StandardScaler().fit_transform(df[METRICS].values)
    f, p, ssa, ssw = _permanova_custom(squareform(pdist(X)), df.genre.values, n_perm=4999, seed=SEED)
    disp, _ = dispersion_check_full(X, df.genre.values)
    return {"encoder": encoder, "pseudo_F": f, "p": p, "SS_between": ssa, "SS_within": ssw,
            "R2": ssa / (ssa + ssw), "Dispersion_F": disp["dispersion_F"], "Dispersion_p": disp["dispersion_p"]}


def classification(df, encoder):
    """10-fold stratified cross-validation (the folds of the ablation analysis)."""
    y = df.is_flash.values
    rf = lambda: make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced"))
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    length = df[["T", "wps_traj"]].values
    five = df[METRICS].values
    sets = [("majority class", "-", lambda: DummyClassifier(strategy="most_frequent"), five),
            ("length (trajectory sentences, words per sentence)", "random forest", rf, length),
            ("length (trajectory sentences, words per sentence)", "logistic regression", lr, length),
            ("five measures", "random forest", rf, five),
            ("five measures", "logistic regression", lr, five)]
    folds = list(StratifiedKFold(10, shuffle=True, random_state=SEED).split(five, y))
    rows = []
    for name, clf_name, make, X in sets:
        acc, f1, auc = [], [], []
        for tr, te in folds:
            model = make().fit(X[tr], y[tr])
            acc.append(accuracy_score(y[te], model.predict(X[te])))
            f1.append(f1_score(y[te], model.predict(X[te])))
            auc.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))
        rows.append({"encoder": encoder, "features": name, "classifier": clf_name,
                     "accuracy": np.mean(acc), "accuracy_sd": np.std(acc), "F1": np.mean(f1), "F1_sd": np.std(f1),
                     "AUROC": np.mean(auc), "AUROC_sd": np.std(auc)})
    return rows


def spectral_window(df):
    """The spectrum uses a 64-point window: shorter series of moves are zero-padded, longer ones use their first 64 moves."""
    steps = df["T"] - 1
    return pd.DataFrame([{"genre": g, "documents": int((df.genre == g).sum()),
                          "step_series_longer_than_64": int(((df.genre == g) & (steps > 64)).sum()),
                          "zero_padded": int(((df.genre == g) & (steps < 64)).sum())} for g in GENRES])


def metric_relations(df, encoder):
    rho = df[METRICS].corr(method="spearman")
    off = rho.values[~np.eye(5, dtype=bool)]
    return {"encoder": encoder, "rho_delta_resonance": rho.loc["delta", "global_resonance"],
            "max_abs_rho": np.abs(off).max(),
            "max_pair": " / ".join(max(((a, b) for a in METRICS for b in METRICS if a < b), key=lambda p: abs(rho.loc[p])))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    save = lambda t, name: (pd.DataFrame(t).to_csv(os.path.join(args.out_dir, name), index=False), pd.DataFrame(t))[1]
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)

    data = {enc: load_encoder(files) for enc, files in ENCODERS.items()}
    main_enc = next(iter(ENCODERS))
    ref = data[main_enc]
    for enc, df in data.items():  # the same documents and sentences under every encoder
        assert (df[["genre", "pkl_index", "T"]].values == ref[["genre", "pkl_index", "T"]].values).all(), enc

    print("== length_parity"); print(save(length_parity(ref), "length_parity.csv").round(4).to_string(index=False))
    print("\n== spectral_window"); print(save(spectral_window(ref), "spectral_window.csv").to_string(index=False))

    es = pd.concat([effect_sizes(df, enc) for enc, df in data.items()])
    print("\n== effect_sizes"); print(save(es, "effect_sizes.csv").round(4).to_string(index=False))
    print("\n== permanova"); print(save([permanova(df, enc) for enc, df in data.items()], "permanova.csv").round(4).to_string(index=False))
    print("\n== metric_relations"); print(save([metric_relations(df, enc) for enc, df in data.items()], "metric_relations.csv").round(3).to_string(index=False))

    cl = [r for enc, df in data.items() for r in classification(df, enc)]
    bow = json.load(open(WORD_BASELINE))
    assert bow["n_flash_fiction"] == int(ref.is_flash.sum()) and bow["n_prose_poem"] == int((1 - ref.is_flash).sum())
    for name in ["random_forest", "logistic_regression"]:
        r = bow[name]
        cl.append({"encoder": "-", "features": "words (TF-IDF)", "classifier": name.replace("_", " "),
                   "accuracy": r["accuracy"]["mean"], "accuracy_sd": r["accuracy"]["sd"], "F1": r["f1"]["mean"],
                   "F1_sd": r["f1"]["sd"], "AUROC": r["roc_auc"]["mean"], "AUROC_sd": r["roc_auc"]["sd"]})
    print("\n== classification"); print(save(cl, "classification.csv").round(3).to_string(index=False))
    inputs = sorted({p for f in ENCODERS.values() for p in f} | {WORD_BASELINE})
    info = {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__, "seed": SEED, "bootstrap_resamples": N_BOOT,
            "scripts_md5": {p: md5(p) for p in [os.path.abspath(__file__).replace(os.getcwd() + os.sep, ""),
                                                "code/measures_and_statistics.py"]},
            "inputs_md5": {p: md5(p) for p in inputs}}
    json.dump(info, open(os.path.join(args.out_dir, "run_info.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
