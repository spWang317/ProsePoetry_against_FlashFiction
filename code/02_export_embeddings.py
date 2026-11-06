# -*- coding: utf-8 -*-
"""Sentence embeddings in the form that is released.

Each text's sentence embeddings (T x d) are multiplied by an orthogonal matrix drawn at random for that text.
An orthogonal transformation keeps every distance, angle and cosine similarity between the sentences of a text,
so the five measures are unchanged, while the vectors no longer lie in the encoder's embedding space.
The matrices are drawn from operating-system entropy and are not stored.

The script needs the encoder outputs written by requires_source_texts/embed_sample.py, which are not included.
Run from the repository root:
    python code/02_export_embeddings.py --in_dir <folder with the encoder outputs> --out_dir data/embeddings
"""
import os, glob, pickle, argparse
import numpy as np


def random_orthogonal(d, rng):
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng()          # operating-system entropy: the transformations cannot be regenerated
    for path in sorted(glob.glob(os.path.join(args.in_dir, "*_embs_*.pkl"))):
        d = pickle.load(open(path, "rb"))
        worst = 0.0
        out = []
        for e in d["embeddings"]:
            e64 = np.asarray(e, dtype=np.float64)
            r = (e64 @ random_orthogonal(e64.shape[1], rng)).astype(np.float32)
            worst = max(worst, float(np.abs(r.astype(np.float64) @ r.astype(np.float64).T - e64 @ e64.T).max()))
            out.append(r)
        d["embeddings"] = out
        pickle.dump(d, open(os.path.join(args.out_dir, os.path.basename(path)), "wb"))
        print(f"{os.path.basename(path)}: {len(out)} texts, largest change in a sentence-pair similarity {worst:.1e}", flush=True)


if __name__ == "__main__":
    main()
