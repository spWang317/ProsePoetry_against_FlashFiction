# -*- coding: utf-8 -*-
import os, re, pandas as pd, numpy as np
from tqdm import tqdm
from scipy.stats import mannwhitneyu

# Check for KSS availability
try:
    import kss
    HAS_KSS = True
except ImportError:
    HAS_KSS = False

# ==========================================
# 1. Standard Preprocessing Functions
# ==========================================

def split_sentences_kss_robust(text):
    """Robust sentence splitting using KSS with regex fallback."""
    if not isinstance(text, str) or not text.strip(): return []
    t = text.strip()
    if HAS_KSS:
        try: sents = kss.split_sentences(t)
        except: sents = re.split(r'(?<=[.!?])\s+|(?<=다\.)\s+|[\n\r]+', t)
    else:
        sents = re.split(r'(?<=[.!?])\s+|(?<=다\.)\s+|[\n\r]+', t)
    return [s.strip() for s in sents if len(s.strip()) > 2]

def preprocess_double_wash(text):
    """
    Standardized Double-Wash logic:
    1. Pre-cleaning (Raw text regex)
    2. Segmentation (KSS)
    3. Post-cleaning (Per-sentence regex + length filter)
    """
    raw_text = str(text)
    # Step 1: Pre-wash raw text (Preserve punctuation for KSS)
    clean_t = re.sub(r'[^가-힣0-9\s.\?!\.,]', ' ', raw_text)
    # Step 2: Split into sentences
    sents = split_sentences_kss_robust(clean_t)
    # Step 3: Post-wash individual sentences
    cleaned_sents = []
    for s in sents:
        s_cleaned = re.sub(r'[^가-힣0-9\s.\?!,]', ' ', s)
        s_cleaned = re.sub(r'\s+', ' ', s_cleaned).strip()
        if len(s_cleaned) > 2:
            cleaned_sents.append(s_cleaned)
    return cleaned_sents

# ==========================================
# 2. Matching Logic
# ==========================================

def stratified_match(df_a, df_b, seed=42, s_step=2, d_step=1.5, min_per_bin=2):
    """Stratified matching based on sentence count and tokens per sentence."""
    rng = np.random.default_rng(seed)
    a, b = df_a.copy(), df_b.copy()
    
    # Define bins for stratification
    s_bins = np.arange(np.floor(min(a["n_sentences"].min(), b["n_sentences"].min())), 
                       np.ceil(max(a["n_sentences"].max(), b["n_sentences"].max())) + s_step, s_step)
    d_bins = np.arange(np.floor(min(a["tokens_per_sent"].min(), b["tokens_per_sent"].min())), 
                       np.ceil(max(a["tokens_per_sent"].max(), b["tokens_per_sent"].max())) + d_step, d_step)

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
    
    return pd.concat(a_out, ignore_index=True), pd.concat(b_out, ignore_index=True)

# ==========================================
# 3. Main Pipeline
# ==========================================

def main():
    base_path = "."
    df_flash = pd.read_csv(os.path.join(base_path, "Flashfiction_0024.csv")).fillna("")
    df_prose = pd.read_csv(os.path.join(base_path, "Prosepoem_0024.csv")).fillna("")

    # Map target columns
    targets = [(df_flash, "text", "FlashFiction"), (df_prose, "text", "ProsePoem")]
    processed_dfs = []
    
    for df, col, name in targets:
        n_sents, tps = [], []
        for text in tqdm(df[col], desc=f"Matching {name}"):
            # Apply standardized Double-Wash
            cleaned_sents = preprocess_double_wash(text)
            n = len(cleaned_sents)
            avg_t = sum(len(s.split()) for s in cleaned_sents) / n if n > 0 else 0
            n_sents.append(n)
            tps.append(avg_t)
        
        df["n_sentences"] = n_sents
        df["tokens_per_sent"] = tps
        # Threshold: documents must have at least 5 sentences
        processed_dfs.append(df[df["n_sentences"] >= 5].copy())

    # Perform matching
    df_f_final, df_p_final = stratified_match(processed_dfs[0], processed_dfs[1])

    # Save matched datasets
    df_f_final.to_csv(os.path.join(base_path, "FlashFiction_final.csv"), index=False, encoding="utf-8-sig")
    df_p_final.to_csv(os.path.join(base_path, "ProsePoem_final.csv"), index=False, encoding="utf-8-sig")
    print(f"🚀 Matching Complete. N={len(df_f_final)} per genre.")

if __name__ == "__main__":
    main()
