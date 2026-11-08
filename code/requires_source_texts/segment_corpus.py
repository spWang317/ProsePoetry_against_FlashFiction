# -*- coding: utf-8 -*-
"""Sentence segmentation of the texts of the corpus and their sentence counts.

Needs the two source files (CSV with the full texts). They are under copyright and are not included.

    python code/requires_source_texts/segment_corpus.py --ff_csv <flash fiction> --pp_csv <prose poems> \
        --rows data/corpus_table.csv --out <folder> --private_dir <folder>

--rows limits the run to the source rows listed in that file (columns genre and row); without it every row of the
source files is segmented.

Every text is split into sentences twice with KSS:
    source-text split   the text as it is; its sentence count and words per sentence are used for the length matching
    trajectory split    symbols other than Hangul, digits, whitespace and . ? ! , are replaced by spaces before
                        splitting; these sentences are embedded
After splitting, the same symbols are replaced by spaces within each sentence, and sentences of two characters or
fewer are dropped.

Outputs
    <out>/corpus_table.csv             one row per text: sentence counts and words per sentence under both splits,
                                       bibliographic columns (no text)
    <out>/segmentation_record.json     KSS version and backend, use of the fallback split
    <private_dir>/segments_<genre>.pkl sentence lists of every text (text; read by embed_sample.py)
"""
import argparse, functools, json, multiprocessing as mp, os, pickle, re, sys, time
import pandas as pd

GENRES = ["ProsePoem", "FlashFiction"]
TEXT_COL = {"ProsePoem": "원문", "FlashFiction": "text"}
BIBLIO = {  # column of the source file -> column of corpus_table.csv
    "ProsePoem": {"책 제목": "book_title", "작품 제목": "work_title", "작가": "author", "출판사": "publisher",
                  "출판년도": "year", "시리즈": "series", "번호": "series_number"},
    "FlashFiction": {"책이름": "book_title", "title": "work_title", "작가": "author", "출판사": "publisher",
                     "출판년도": "year", "isbn": "isbn", "국가": "nationality"},
}
COLUMNS = ["genre", "row", "n_sent_source", "wps_source", "n_sent_traj", "wps_traj",
           "book_title", "work_title", "author", "publisher", "year", "series", "series_number", "isbn", "nationality"]
FALLBACK_RX = r'(?<=[.!?])\s+|(?<=다\.)\s+|[\n\r]+'      # used only if KSS raises an exception


# ---------------------------------------------------------------- segmentation
def _split(t, backend):
    import kss
    try:
        return kss.split_sentences(t, backend=backend), 0
    except Exception:
        return re.split(FALLBACK_RX, t), 1


def _clean_sentences(sents):
    out = []
    for s in sents:
        s = s.strip()
        if len(s) <= 2:
            continue
        c = re.sub(r'\s+', ' ', re.sub(r'[^가-힣0-9\s.\?!,]', ' ', s)).strip()
        if len(c) > 2:
            out.append(c)
    return out


def words_per_sentence(sents):
    return sum(len(s.split()) for s in sents) / len(sents) if sents else 0.0


def segment(text, backend="auto"):
    """Sentence lists of one text under the two splits, and whether the fallback split was used."""
    raw = str(text)
    pre = re.sub(r'[^가-힣0-9\s.\?!,]', ' ', raw).strip()        # trajectory split: other characters replaced by spaces before splitting
    t = raw.strip() if isinstance(text, str) else ""             # source-text split: the text as it is
    if pre:
        s, f_traj = _split(pre, backend)
        traj = _clean_sentences(s)
    else:
        traj, f_traj = [], 0
    if not t:
        source, f_source = [], 0
    elif t == pre:                                               # identical input, identical result
        source, f_source = traj, f_traj
    else:
        s, f_source = _split(t, backend)
        source = _clean_sentences(s)
    return {"traj": traj, "source": source, "fallback_traj": f_traj, "fallback_source": f_source}


def segment_texts(texts, workers, backend, label):
    t0, fn = time.time(), functools.partial(segment, backend=backend)
    with mp.get_context("spawn").Pool(workers) as pool:
        segs = pool.map(fn, texts, chunksize=4)
    print(f"  [{label}] segmented {len(segs)} texts ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return segs


# ---------------------------------------------------------------- corpus table
def genre_table(genre, df, private_dir, workers=8, backend="auto"):
    """df: the source rows of one genre; its index is the identifier of the text (`row`).
    Returns the rows of corpus_table.csv for the genre and the number of texts split with the fallback."""
    rows = df.index.tolist()
    segs = segment_texts(df[TEXT_COL[genre]].fillna("").tolist(), workers, backend, genre)
    with open(os.path.join(private_dir, f"segments_{genre}.pkl"), "wb") as f:
        pickle.dump({"rows": rows, "source": [s["source"] for s in segs], "traj": [s["traj"] for s in segs]}, f)

    t = pd.DataFrame({"genre": genre, "row": rows,
                      "n_sent_source": [len(s["source"]) for s in segs],
                      "wps_source": [words_per_sentence(s["source"]) for s in segs],
                      "n_sent_traj": [len(s["traj"]) for s in segs],
                      "wps_traj": [words_per_sentence(s["traj"]) for s in segs]})
    for src, dst in BIBLIO[genre].items():
        t[dst] = df[src].values
    for c in ["year", "series_number", "isbn"]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce").astype("Int64")
    fallback = {"source_split": int(sum(s["fallback_source"] for s in segs)),
                "trajectory_split": int(sum(s["fallback_traj"] for s in segs))}
    return t, fallback


def build(frames, out, private_dir, workers=8, backend="auto"):
    """frames: genre -> DataFrame of source rows. Writes corpus_table.csv and segmentation_record.json to `out`."""
    os.makedirs(out, exist_ok=True)
    os.makedirs(private_dir, exist_ok=True)
    import kss
    from kss._utils.sanity_checks import _check_analyzer_backend
    record = {"python": sys.version.split()[0], "kss": kss.__version__,
              "kss_backend": str(_check_analyzer_backend(backend)._backend),
              "texts": {}, "texts_split_with_fallback": {}}
    print("versions:", {k: record[k] for k in ["python", "kss", "kss_backend"]}, flush=True)
    tables = []
    for genre in GENRES:
        t, fallback = genre_table(genre, frames[genre], private_dir, workers, backend)
        record["texts"][genre], record["texts_split_with_fallback"][genre] = len(t), fallback
        tables.append(t)
        print(f"{genre}: {len(t)} texts | at least five sentences under both splits "
              f"{int(((t.n_sent_source >= 5) & (t.n_sent_traj >= 5)).sum())} | fallback split {fallback}", flush=True)
    pd.concat(tables, ignore_index=True).reindex(columns=COLUMNS).to_csv(os.path.join(out, "corpus_table.csv"), index=False)
    json.dump(record, open(os.path.join(out, "segmentation_record.json"), "w"), indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ff_csv", required=True)
    ap.add_argument("--pp_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--private_dir", required=True, help="folder for the sentence lists (text)")
    ap.add_argument("--kss_backend", default="auto")
    ap.add_argument("--workers", type=int, default=max(1, min(32, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--rows", help="CSV with the columns genre and row (data/corpus_table.csv): only these source rows are segmented")
    args = ap.parse_args()
    paths = {"FlashFiction": args.ff_csv, "ProsePoem": args.pp_csv}
    frames = {g: pd.read_csv(paths[g], low_memory=False) for g in GENRES}
    if args.rows:
        keep = pd.read_csv(args.rows, usecols=["genre", "row"])
        frames = {g: f.loc[sorted(keep.loc[keep.genre == g, "row"].astype(int))] for g, f in frames.items()}
    build(frames, args.out, args.private_dir, args.workers, args.kss_backend)


if __name__ == "__main__":
    main()
