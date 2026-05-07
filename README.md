<div align="center">
  <h1>🌟 BhojRAG</h1>
  <p><b>Robust Retrieval-Augmented Generation for Unstandardized Low-Resource Indic Languages</b></p>
  
  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Status](https://img.shields.io/badge/Status-Research_Prototype-success.svg)]()
</div>

<br/>

> **BhojRAG** bridges the NLP gap for **Bhojpuri** (~50 million speakers) by addressing severe semantic scarcity and orthographic inconsistency using a novel **Character N-gram BM25 + Fine-tuned MuRIL Hybrid Pipeline**.

---

## 📖 Overview

Retrieval-Augmented Generation (RAG) systems have shown significant promise in grounding LLM outputs. However, their effectiveness collapses for low-resource languages with high spelling variation. 

**BhojRAG** solves this for Bhojpuri — a language lacking standardized orthography — by introducing a hybrid architecture:
1. **Character n-gram BM25:** Robust to arbitrary spelling variations (e.g., *भोजपूरी* vs *भोजपुरी*).
2. **Fine-Tuned MuRIL Embeddings:** Adapted via contrastive learning on synthetic Bhojpuri QA pairs.
3. **Reciprocal Rank Fusion (RRF):** Fusing sparse and dense methods to achieve state-of-the-art complementary coverage.

## 🚀 Key Contributions

- 🔠 **Orthographic Robustness:** A custom sparse retriever tokenizing at the character n-gram level to capture subword overlaps between spelling variants.
- 🧠 **Zero-Resource Contrastive Fine-Tuning:** Fine-tuning MuRIL on synthetically generated QA pairs using `MultipleNegativesRankingLoss`.
- 🌉 **Hindi-Bridge Prompting:** A novel RAG strategy leveraging Hindi as a reasoning bridge for Bhojpuri.
- 🔬 **Reproducible Framework:** A highly modular, config-driven pipeline built for rigorous ablation studies and extensible to other Indic languages.

---

## 📊 Evaluation & Results

Our system demonstrates that **character-level sparse retrieval significantly outperforms word-level baselines** for unstandardized text, and hybrid fusion yields the best overall retrieval quality.

### Retrieval Comparison

| System | Description |
|--------|-------------|
| **Word BM25** | Standard Okapi BM25 (whitespace tokenization) |
| **Char N-gram BM25** | Custom char n-gram BM25 with boundary markers |
| **Zero-shot MuRIL** | Out-of-the-box multilingual embeddings |
| **Fine-tuned MuRIL** | Contrastively trained on synthetic pairs |
| **Hybrid (RRF)** | N-gram BM25 + Fine-tuned MuRIL fusion |

<div align="center">
  <img src="paper_assets/figures/retrieval_comparison.png" alt="Retrieval Comparison" width="70%"/>
</div>

### N-gram Ablation Study

Determining the optimal n-gram range is critical for handling Devanagari sub-word units effectively.

<div align="center">
  <img src="paper_assets/figures/ablation_heatmap.png" alt="Ablation Heatmap" width="70%"/>
</div>

---

## ⚙️ Quick Start

> [!IMPORTANT]
> To run the complete RAG pipeline including the generation step, ensure you have your LLM API keys exported (e.g., `OPENAI_API_KEY` or `GOOGLE_API_KEY`). If no key is found, the system gracefully falls back to a **retrieval-only** mode.

### 1. Installation

```bash
git clone https://github.com/im-anishraj/BhojRAG.git
cd BhojRAG
python -m venv venv
# Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

The project is driven by `configs/default.yaml` and separated into highly logical, reproducible numbered scripts.

```bash
# Step 1: Preprocess, clean, and chunk the raw Bhojpuri corpus
python scripts/01_prepare_data.py --config configs/default.yaml

# Step 2: Generate synthetic QA pairs using template techniques
python scripts/02_generate_qa.py --config configs/default.yaml

# Step 3: Contrastively fine-tune the dense MuRIL retriever (GPU Recommended)
python scripts/03_train_dense.py --config configs/default.yaml

# Step 4: Build BM25 and FAISS Indices
python scripts/04_build_indices.py --config configs/default.yaml

# Step 5: Evaluate all variants (Use --sparse-only to skip dense models)
python scripts/05_evaluate.py --config configs/default.yaml --sparse-only

# Step 6: Test End-to-End RAG Inference
python scripts/06_run_rag.py --config configs/default.yaml --query "भोजपुरी के इतिहास का ह?"

# Step 7: Generate Publication Assets (LaTeX tables & PNG plots)
python scripts/07_generate_paper_assets.py --config configs/default.yaml
```

---

## 📁 Repository Architecture

```text
BhojRAG/
├── configs/               # YAML experiment configurations
├── data/                  # Raw corpus, processed chunks, synthetic QA
├── models/                # FAISS indices & saved model checkpoints
├── paper_assets/          # Generated LaTeX tables and Matplotlib figures
├── scripts/               # Sequenced execution scripts (01 to 07)
├── src/
│   ├── data/              # Ingestion, transliteration, chunking
│   ├── eval/              # MRR/NDCG metrics, Multi-retriever Benchmarking
│   ├── rag/               # LLM Generation, Prompts, Backends
│   ├── retrieval/         # N-gram BM25, MuRIL Dense, RRF Hybrid
│   └── utils/             # Config parsing, MLflow logging, reproducibility
└── tests/                 # Comprehensive Pytest suite
```

---

## 🎯 Target Venues

This framework is built for academic submission. Potential targets include:
- **ACL / EMNLP / NAACL** (Findings track)
- **LREC-COLING**
- **Workshop:** AfricaNLP / LowResourceNLP / VarDial

## 📝 License & Citation

This project is licensed under the **MIT License**.

```bibtex
@misc{bhojrag2026,
  title={BhojRAG: Robust Retrieval-Augmented Generation for Unstandardized Low-Resource Indic Languages},
  author={Anish Raj},
  year={2026},
  howpublished={\url{https://github.com/im-anishraj/BhojRAG}}
}
```
<div align="center">
  <i>Built with ❤️ for Low-Resource Indic NLP</i>
</div>
