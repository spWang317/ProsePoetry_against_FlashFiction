# Semantic Dynamics: 5D Framework Analysis

This repository contains the official implementation of the **5D Semantic Dynamics Framework**. The project analyzes structural differences between Flash Fiction and Prose Poetry using high-dimensional semantic trajectories and advanced statistical diagnostics.

> **Note on Data Availability**: Due to copyright restrictions, the original text corpora (`.csv`) are not provided in this repository. Instead, we provide **pre-computed semantic embeddings (`.pkl`)** and comprehensive metadata to ensure full reproducibility for reviewers and researchers without violating copyright laws.

---

## ⚡ Quick Start (For Reviewers & Reproduction)

If you wish to reproduce the statistical results and figures using the provided embedding files, follow these steps in your local environment.

### 1. Environment Setup
```bash
# Create a local environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install analysis-specific dependencies
pip install -r requirements.txt
```

### 2. Execution Sequence
Run the following scripts in order. The analysis will use the pre-computed `.pkl` files in the root directory.
```bash
# Step 1: Run statistical analysis (Outputs: CSV tables and logs)
python3 core_statistical_validation_5d_for_emb.py

# Step 2: Generate publication-quality figures
python3 statistical_visualization.py
```
*The results will be saved in `./Final_Research_Output_5D_Restored` and `./Figures_Final_5D_Hardened`.*

---

## 🛠 Repository Structure & Pipeline

The pipeline is divided into two distinct phases to separate heavy data extraction from lightweight statistical reproduction.

### Phase 1: Semantic Extraction (Author-Side / Server Only)
This phase requires access to the raw text corpora and high-performance GPU resources. 
1. `stratified_length_matching.py`: Preprocesses raw texts and synchronizes length distributions between genres to eliminate confounding variables.
2. `text_to_pkl.py`: Extracts S-RoBERTa embeddings (using `ko-sroberta-multitask`) and saves them as numerical trajectories in `.pkl` format.

### Phase 2: Statistical Validation & Visualization (Public / Local)
This phase uses the numerical embedding data to validate the 5D framework.
1. `core_statistical_validation_5d_for_emb.py`: Calculates geometric metrics (Displacement, Volatility, Resonance, Curvature, and Spectral Centroid). Performs OLS regression, PERMANOVA, and Ablation studies.
2. `statistical_visualization.py`: Produces structural validation heatmaps, genre-effect plots, radar profiles, and bivariate phase-space distributions.

---

## 💻 System Specifications (Author's Extraction Environment)

The initial embedding extraction was performed on a high-performance computing (HPC) server to handle the deep learning modeling:

* **Operating System**: Ubuntu 24.04.2 LTS
* **CPU**: Intel(R) Xeon(R) Silver 4516Y+
* **GPU**: 2 x NVIDIA RTX A6000 (48GB VRAM each)
* **RAM**: 251 GiB
* **Python Version**: 3.10+

*Note: Phase 2 (Analysis & Visualization) is hardware-agnostic and can be performed on a standard laptop (8GB+ RAM recommended).*

---

## 📖 Data Availability & Metadata

### Copyright Restrictions
* `Flashfiction_0024.csv` & `Prosepoem_0024.csv`: **Not Publicly Available** due to copyright protection of the literary works.

### Reproducibility Package
* `FlashFiction_embs.pkl` & `ProsePoem_embs.pkl`: **Not provided due to file size limitations on GitHub.** High-dimensional semantic vectors that serve as the numerical foundation for all findings.
* `Flashfiction_book_list.csv` & `Prosepoem_book_list.csv`: Comprehensive metadata including book titles, authors, publishers, publication years, and ISBNs to allow for independent corpus verification.


## 📄 Citation & Preprint

If you find this framework or the provided semantic dynamics analysis useful for your research, please cite our preprint:

**"From Kinetic to Resonant Regimes: Geometric Differentiation of Prose Poetry against Flash Fiction"**

* **Preprint Server**: Research Square
* **DOI**: [10.21203/rs.3.rs-8722451/v1](https://doi.org/10.21203/rs.3.rs-8722451/v1)
* **URL**: [https://www.researchsquare.com/article/rs-8722451/v1](https://www.researchsquare.com/article/rs-8722451/v1)
