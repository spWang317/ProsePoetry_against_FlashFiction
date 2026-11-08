# Geometric Differentiation of Korean Prose Poetry against Flash Fiction

Code and data for the article *Geometric Differentiation of Korean Prose Poetry against Flash Fiction*.

The study compares Korean prose poems and flash fiction of overlapping length. Each text is treated as the sequence of its sentence embeddings (a trajectory), and the trajectory is described by five measures. The same sentences are embedded with three encoders.

The source texts are under copyright and are not included. Everything from the sentence embeddings onwards can be reproduced with the files in this repository.

## Contents

```
data/
  corpus_table.csv         one row per text of the corpus: sentence counts, bibliographic record
  sample_rows.csv          the analysis sample: 379 flash-fiction texts and 379 prose poems (written by code/01)
  embeddings/              sentence embeddings of the analysis sample under three encoders, and the result
                           of the word-level baseline (word_baseline.json)
  segmentation_record.json, embedding_record.json
                           records written by the two scripts that need the source texts
code/
  requires_source_texts/
    segment_corpus.py      sentence segmentation of the texts of the corpus; writes data/corpus_table.csv
    embed_sample.py        sentence embeddings of the analysis sample under three encoders, word-level baseline
  01_length_matching.py    length matching of the two genres
  02_export_embeddings.py  transformation applied to the encoder outputs (needs the outputs of embed_sample.py,
                           which are not included)
  measures_and_statistics.py  the five measures and the per-encoder statistics
  03_tables.py             tables reported in the article
  04_distribution_statistics.py
  05_sample_composition.py
  06_figures.py
  07_words_per_sentence_check.py  sensitivity check: trajectory words per sentence as a second covariate (Supplementary Table S7)
outputs/                   output of the scripts above, as reported in the article
environment/               software environment of the segmentation and embedding stage
requirements.txt           software environment of the analysis
LICENSE                    licence of the code (MIT)
LICENSE-DATA               licence of the data and outputs (CC BY 4.0)
```

## Reproducing the results

The analysis runs in this order: `segment_corpus.py` → `01` → `embed_sample.py` → `02` → `measures_and_statistics.py` → `03` to `07`. The two scripts in `requires_source_texts/` and `02_export_embeddings.py` need files that are not included; their outputs are in `data/`. The other steps run on the files of this repository.

Python 3.12 with the packages in `requirements.txt` (the embedding files need NumPy 2). Run from the repository root.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python code/01_length_matching.py > outputs/length_matching.txt

E=data/embeddings
python code/measures_and_statistics.py --ff_pkl $E/FlashFiction_embs_ko-sroberta-multitask.pkl --pp_pkl $E/ProsePoem_embs_ko-sroberta-multitask.pkl --out_dir outputs/statistics/ko-sroberta-multitask
python code/measures_and_statistics.py --ff_pkl $E/FlashFiction_embs_bge-m3.pkl --pp_pkl $E/ProsePoem_embs_bge-m3.pkl --out_dir outputs/statistics/bge-m3
python code/measures_and_statistics.py --ff_pkl $E/FlashFiction_embs_me5-large.pkl --pp_pkl $E/ProsePoem_embs_me5-large.pkl --out_dir outputs/statistics/multilingual-e5-large

python code/03_tables.py --out_dir outputs/tables > outputs/tables_log.txt
python code/04_distribution_statistics.py
python code/05_sample_composition.py > outputs/sample_composition.txt
python code/06_figures.py --out_dir outputs/figures
python code/07_words_per_sentence_check.py > outputs/words_per_sentence_check.txt
```

The scripts overwrite the files in `outputs/`. The bootstrap intervals in `03_tables.py` take about one minute.

| Reported in the article | File |
|---|---|
| Stages of the matching, composition of the sample | `outputs/length_matching.txt`, `outputs/sample_composition.txt` |
| Lengths of the two genres | `outputs/tables/length_parity.csv` |
| Genre differences under three encoders (medians, Cliff's delta with bootstrap interval, Mann-Whitney U, length-controlled regression) | `outputs/tables/effect_sizes.csv` |
| Multivariate comparison, variance explained | `outputs/tables/permanova.csv`, `outputs/statistics/*/Dispersion_Check_5D.csv` |
| Classification of single texts | `outputs/tables/classification.csv` |
| Texts whose series of move lengths exceeds the 64-point window | `outputs/tables/spectral_window.csv` |
| Sentences longer than the input limit of an encoder, use of the fallback sentence split | `data/embedding_record.json`, `data/segmentation_record.json` |
| Correlations between the measures, variance inflation factors | `outputs/statistics/*/Metric_Orthogonality_Spearman.csv`, `Metric_VIF_5D.csv`, `outputs/tables/metric_relations.csv` |
| Ablation | `outputs/statistics/*/Ablation_Study_Results_5D.csv` (`Drop_*` = all five measures minus the reduced set; positive values are a loss) |
| Distribution statistics | `outputs/statistics/ko-sroberta-multitask/Distribution_Statistics.csv` |
| Figures | `outputs/figures/` |
| Genre coefficients with trajectory words per sentence as a second covariate (Supplementary Table S7) | `outputs/words_per_sentence_check.csv`, `outputs/words_per_sentence_check.txt` |

`outputs/statistics/<encoder>/` also holds the measures of every text (`Pure_Metrics_Data_Final_5D.csv`; `n_sent` is the trajectory sentence count and `n_sent_matching` the source-text sentence count), the per-measure tests (`Final_Summary_Table_5D.csv`), the multivariate test (`PERMANOVA_Result_5D.txt`) and a log (`Stability_Log.txt`).

## Data

### `data/corpus_table.csv`

One row per text of the corpus (2,795 flash-fiction texts, 5,131 prose poems). `row` is the row of the text in its source file; source rows that are not part of the corpus do not appear in the table. The sentence counts are written by `requires_source_texts/segment_corpus.py --rows data/corpus_table.csv`, which segments the listed rows; the bibliographic columns identify the text (book, work, author, publisher, year, series number, ISBN).

| Column | Meaning |
|---|---|
| `genre`, `row` | `FlashFiction` or `ProsePoem`, and the identifier of the text |
| `n_sent_source`, `wps_source` | number of sentences and mean words per sentence when the source text is split as it is; used for matching. After splitting, characters other than Hangul, digits, whitespace and `. ? ! ,` are replaced by spaces in each sentence and sentences of two characters or fewer are dropped |
| `n_sent_traj`, `wps_traj` | the same when characters other than Hangul, digits, whitespace and `. ? ! ,` are replaced by spaces before splitting; these sentences are embedded |
| `book_title`, `work_title`, `author`, `publisher`, `year` | bibliographic record |
| `series`, `series_number` | poetry series and volume number (prose poems) |
| `isbn`, `nationality` | ISBN of the book and nationality of the author (flash fiction) |

### `data/sample_rows.csv`

`01_length_matching.py` matches the texts that have at least five sentences under both splits on the source-text sentence count and words per sentence. The analysis sample is 379 texts per genre. `row` refers to `corpus_table.csv`.

### `data/embeddings/`

`<Genre>_embs_<encoder>.pkl` is a pickled dictionary with one entry per text of the analysis sample, in the order of `data/sample_rows.csv`.

| Key | Meaning |
|---|---|
| `embeddings` | list of arrays of shape (trajectory sentences, dimension), one row per sentence, unit length |
| `n_sentences`, `tokens_per_sent` | source-text sentence count and words per sentence (the matching variables) |
| `n_sentences_traj`, `tokens_per_sent_traj` | sentence count and words per sentence of the embedded trajectory |
| `pkl_index`, `corpus_row` | position in the sample and row in `corpus_table.csv` |

Encoders: `jhgan/ko-sroberta-multitask` (768 dimensions), `BAAI/bge-m3` (1,024), `intfloat/multilingual-e5-large` (1,024; file tag `me5-large`).

**The embeddings of each text have been multiplied by an orthogonal matrix drawn at random for that text** (`02_export_embeddings.py`). This keeps every distance, angle and cosine similarity between the sentences of a text, so the five measures and all statistics are reproduced. The transformation is designed to prevent the copyrighted sentences from being reconstructed from the vectors. The matrices were not stored. As a consequence the vectors of different texts cannot be compared with each other, and they are not in the embedding space of the encoder.

`word_baseline.json` holds the cross-validated scores of the word-level classifier (TF-IDF of the words of the trajectory sentences, logistic regression). It is written by `requires_source_texts/embed_sample.py`.

## The five measures

Computed in `code/measures_and_statistics.py` (`compute_5d_metrics`) from the unit-length sentence embeddings e_1, ..., e_T of a text.

| Measure | Column in the output files | Definition |
|---|---|---|
| Displacement | `delta` | Euclidean distance between the last and the first sentence |
| Transition volatility | `surprisal_std` | standard deviation of s_t = -ln((cos(e_t, e_{t+1}) + 1) / 2 + 1e-12) over neighbouring sentences |
| Global resonance | `global_resonance` | mean cosine similarity over all pairs of sentences |
| Curvature | `curvature` | mean angle between successive moves e_{t+1} - e_t; angles that involve a move of length 1e-4 or less are left out, and the measure is undefined with fewer than two angles |
| Spectral centroid | `spec_centroid` | power-weighted mean frequency of the periodogram (64 points) of the mean-removed series of move lengths; shorter series are zero-padded and longer series use their first 64 values |

## Stages that need the source texts

- `segment_corpus.py` splits the texts listed in `data/corpus_table.csv` (`--rows`) into sentences under the two splits and writes `data/corpus_table.csv` and `data/segmentation_record.json` (sentence splitter and its backend, number of texts for which the fallback split was used: none).
- `embed_sample.py` embeds the trajectory sentences of the analysis sample with the three encoders at fixed model revisions, counts the sentences longer than the input limit of each encoder, and computes the word-level baseline. It writes the encoder outputs that `02_export_embeddings.py` transforms, `data/embeddings/word_baseline.json` and `data/embedding_record.json`.

Sentence splitting uses KSS 6.0.6 with the Mecab backend through konlpy. With a different backend the sentence boundaries differ. The environment of this stage is recorded in `environment/` (conda export, `pip freeze`, the backend selected by KSS, and the Mecab installation). Mecab (mecab-ko 0.996/ko-0.9.2) and its dictionary (mecab-ko-dic) are system packages and have to be installed separately.

## Software

Analysis: Python 3.12.13, NumPy 2.1.2, pandas 2.3.3, SciPy 1.15.3, statsmodels 0.14.6, scikit-learn 1.7.2, Matplotlib 3.10.8.
Segmentation and embedding: Python 3.10.19, KSS 6.0.6, sentence-transformers 5.2.0, PyTorch 2.9.1.

## Licence

The code (`code/`) is released under the MIT License (`LICENSE`). The data and outputs (`data/`, `outputs/`, `environment/`) are released under the Creative Commons Attribution 4.0 International licence (`LICENSE-DATA`). The source texts remain under the copyright of their authors and publishers and are not part of this repository.
