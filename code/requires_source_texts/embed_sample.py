# -*- coding: utf-8 -*-
"""Sentence embeddings of the analysis sample under three encoders, and the word-level baseline.

Needs the sentence lists written by segment_corpus.py (text; not included).

    python code/requires_source_texts/embed_sample.py --sample data/sample_rows.csv --private_dir <folder> --out <folder>

The trajectory sentences of every sampled text are embedded with the three encoders (unit-length vectors,
one row per sentence). The word-level baseline classifies the genre of a text from the words of its
trajectory sentences (TF-IDF, logistic regression, 10-fold stratified cross-validation).

Outputs in --out (no text)
    <Genre>_embs_<encoder>.pkl   sentence embeddings and lengths of the sampled texts, in the order of the sample
                                 (input of 02_export_embeddings.py)
    word_baseline.json           cross-validated scores of the word-level classifier
    embedding_record.json        software versions, encoder revisions, sentences longer than the input limit
"""
import os, sys, json, time, pickle, argparse, platform
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment_corpus import words_per_sentence

GENRES = ["ProsePoem", "FlashFiction"]
ENCODERS = [  # file tag, model, revision, sentence prefix
    ("ko-sroberta-multitask", "jhgan/ko-sroberta-multitask", "8fca7c9c98c26599be0e14b9916b11a756a26f19", ""),
    ("bge-m3", "BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181", ""),
    ("me5-large", "intfloat/multilingual-e5-large", "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3", "query: "),
]
MAX_TOKENS = 512          # input limit applied to the two multilingual encoders


def embed_docs(model, docs, prefix="", batch_size=128, label=""):
    """docs: list of sentence lists -> list of (n_sent, dim) float32 arrays (unit length)."""
    flat = [prefix + s for d in docs for s in d]
    if not flat:
        return [np.zeros((0, 0), dtype=np.float32) for _ in docs]
    parts, t0, step = [], time.time(), 20000
    for i in range(0, len(flat), step):
        parts.append(model.encode(flat[i:i + step], normalize_embeddings=True,
                                  batch_size=batch_size, show_progress_bar=False))
        print(f"  [{label}] embedded {min(i + step, len(flat))}/{len(flat)} sentences "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    allv = np.vstack(parts).astype(np.float32)
    out, k = [], 0
    for d in docs:
        out.append(allv[k:k + len(d)])
        k += len(d)
    return out


def count_over_limit(model, sentences, prefix):
    """Number of sentences whose token length exceeds the input limit of the encoder."""
    limit, over = int(model.max_seq_length), 0
    for i in range(0, len(sentences), 2000):
        ids = model.tokenizer([prefix + s for s in sentences[i:i + 2000]],
                              add_special_tokens=True, truncation=False, padding=False)["input_ids"]
        over += sum(len(x) > limit for x in ids)
    return {"max_seq_length": limit, "n_sentences": len(sentences), "n_over_limit": int(over)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True, help="sample_rows.csv written by 01_length_matching.py")
    ap.add_argument("--private_dir", required=True, help="folder holding segments_<genre>.pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch, sklearn, sentence_transformers
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.pipeline import make_pipeline

    record = {"python": platform.python_version(), "torch": torch.__version__, "scikit-learn": sklearn.__version__,
              "sentence_transformers": sentence_transformers.__version__, "numpy": np.__version__,
              "encoders": [{"tag": t, "model": m, "revision": r, "sentence_prefix": p} for t, m, r, p in ENCODERS],
              "token_limits": {}}

    sample = pd.read_csv(args.sample)
    docs = {}     # genre -> (sample rows, trajectory sentences of each text)
    for genre in GENRES:
        cache = pickle.load(open(os.path.join(args.private_dir, f"segments_{genre}.pkl"), "rb"))
        at = {r: k for k, r in enumerate(cache["rows"])}
        s = sample[sample.genre == genre].sort_values("sample_index").reset_index(drop=True)
        sent_lists = []
        for r in s.itertuples():
            source, traj = cache["source"][at[int(r.row)]], cache["traj"][at[int(r.row)]]
            if len(source) != int(r.n_sentences) or abs(words_per_sentence(source) - float(r.tokens_per_sent)) > 1e-9 or len(traj) < 5:
                sys.exit(f"[{genre}] row {int(r.row)}: the sentence lists do not agree with the sample file")
            sent_lists.append(traj)
        s["n_sentences_traj"] = [len(x) for x in sent_lists]
        s["tokens_per_sent_traj"] = [words_per_sentence(x) for x in sent_lists]
        docs[genre] = (s, sent_lists)
        print(f"[{genre}] {len(s)} texts, {sum(map(len, sent_lists))} trajectory sentences", flush=True)

    for tag, name, revision, prefix in ENCODERS:
        model = SentenceTransformer(name, device=args.device, revision=revision)
        if tag != "ko-sroberta-multitask":
            model.max_seq_length = min(int(model.max_seq_length or MAX_TOKENS), MAX_TOKENS)
        limits = {"max_seq_length": int(model.max_seq_length), "n_sentences": 0, "n_over_limit": 0}
        for genre in GENRES:
            s, sent_lists = docs[genre]
            embs = embed_docs(model, sent_lists, prefix=prefix, batch_size=args.batch_size, label=f"{genre} {tag}")
            c = count_over_limit(model, [x for sl in sent_lists for x in sl], prefix)
            limits["n_sentences"] += c["n_sentences"]; limits["n_over_limit"] += c["n_over_limit"]
            with open(os.path.join(args.out, f"{genre}_embs_{tag}.pkl"), "wb") as f:
                pickle.dump({"embeddings": embs,
                             "n_sentences": s.n_sentences.astype(int).tolist(),
                             "tokens_per_sent": s.tokens_per_sent.astype(float).tolist(),
                             "n_sentences_traj": s.n_sentences_traj.astype(int).tolist(),
                             "tokens_per_sent_traj": s.tokens_per_sent_traj.astype(float).tolist(),
                             "pkl_index": s.sample_index.astype(int).tolist(),
                             "corpus_row": s.row.astype(int).tolist()}, f)
        record["token_limits"][tag] = limits
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # word-level baseline
    texts = np.array([" ".join(sl) for genre in GENRES for sl in docs[genre][1]], dtype=object)
    y = np.array([1 if genre == "FlashFiction" else 0 for genre in GENRES for _ in docs[genre][1]])
    pipe = make_pipeline(TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\S+", min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=5000))
    cv = cross_validate(pipe, texts, y, cv=StratifiedKFold(10, shuffle=True, random_state=42),
                        scoring=["accuracy", "f1", "roc_auc"])
    bow = {"n_flash_fiction": int(y.sum()), "n_prose_poem": int((1 - y).sum()),
           **{k: {"mean": float(cv[f"test_{k}"].mean()), "sd": float(cv[f"test_{k}"].std()),
                  "folds": cv[f"test_{k}"].tolist()} for k in ["accuracy", "f1", "roc_auc"]}}
    json.dump(bow, open(os.path.join(args.out, "word_baseline.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(record, open(os.path.join(args.out, "embedding_record.json"), "w"), indent=1, ensure_ascii=False)
    print("word-level baseline: " + ", ".join(f"{k} {bow[k]['mean']:.3f} (sd {bow[k]['sd']:.3f})" for k in ["accuracy", "f1", "roc_auc"]))
    print("sentences longer than the input limit:", record["token_limits"])


if __name__ == "__main__":
    main()
