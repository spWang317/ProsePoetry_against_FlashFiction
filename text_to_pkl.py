# -*- coding: utf-8 -*-
import os, re, pickle, pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Re-use preprocessing logic from Step 1 for perfect consistency
def split_sentences_kss_robust(text):
    import re
    try: import kss
    except: kss = None
    t = text.strip()
    if kss:
        try: sents = kss.split_sentences(t)
        except: sents = re.split(r'(?<=[.!?])\s+|(?<=다\.)\s+|[\n\r]+', t)
    else: sents = re.split(r'(?<=[.!?])\s+|(?<=다\.)\s+|[\n\r]+', t)
    return [s.strip() for s in sents if len(s.strip()) > 2]

def preprocess_double_wash(text):
    raw_text = str(text)
    clean_t = re.sub(r'[^가-힣0-9\s.\?!\.,]', ' ', raw_text)
    sents = split_sentences_kss_robust(clean_t)
    cleaned_sents = []
    for s in sents:
        s_cleaned = re.sub(r'[^가-힣0-9\s.\?!,]', ' ', s)
        s_cleaned = re.sub(r'\s+', ' ', s_cleaned).strip()
        if len(s_cleaned) > 2: cleaned_sents.append(s_cleaned)
    return cleaned_sents

def main():
    base_path = "."
    # Load SRoBERTa model optimized for Korean
    model = SentenceTransformer("jhgan/ko-sroberta-multitask", device="cuda")
    
    targets = [
        ("FlashFiction_final.csv", "text", "FlashFiction_embs.pkl"),
        ("ProsePoem_final.csv", "text", "ProsePoem_embs.pkl")
    ]

    for f_in, col, f_out in targets:
        full_in_path = os.path.join(base_path, f_in)
        if not os.path.exists(full_in_path): continue
        
        df = pd.read_csv(full_in_path).fillna("")
        all_embeddings = []
        
        for text in tqdm(df[col], desc=f"Vectorizing {f_in}"):
            # Consistent Double-Wash
            cleaned_sents = preprocess_double_wash(text)
            # Encode with normalization (crucial for cosine similarity metrics)
            embs = model.encode(cleaned_sents, normalize_embeddings=True)
            all_embeddings.append(embs)
            
        # Store metadata along with embeddings
        result_data = {
            "embeddings": all_embeddings,
            "n_sentences": df["n_sentences"].tolist(),
            "tokens_per_sent": df["tokens_per_sent"].tolist()
        }
        
        with open(os.path.join(base_path, f_out), 'wb') as f:
            pickle.dump(result_data, f)
    print("✅ Numerical data (PKL) generation complete.")

if __name__ == "__main__":
    main()
