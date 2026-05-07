# BhojRAG: Robust Retrieval-Augmented Generation for Unstandardized Low-Resource Indic Languages

## Abstract

Retrieval-Augmented Generation (RAG) systems have shown significant promise in grounding large language model outputs, but their effectiveness collapses for low-resource languages with high orthographic variation. **BhojRAG** addresses this gap for **Bhojpuri** — an Indo-Aryan language spoken by ~50 million people across Bihar, UP, Jharkhand, and the global diaspora — which lacks standardized orthography and is severely underrepresented in multilingual NLP resources. We propose a hybrid retrieval architecture combining **character n-gram BM25** (robust to spelling variation) with **fine-tuned dense embeddings** (based on MuRIL, adapted via contrastive learning on synthetic Bhojpuri QA pairs), fused through **Reciprocal Rank Fusion (RRF)**. Our system demonstrates that character-level sparse retrieval significantly outperforms word-level BM25 for unstandardized text, that task-specific dense fine-tuning improves upon zero-shot multilingual encoders, and that hybrid fusion yields the best overall retrieval quality. The architecture is designed to be extensible to other low-resource Indic languages facing similar orthographic challenges.

## Key Contributions

1. **Character n-gram BM25 for orthographic robustness**: A custom sparse retriever that tokenizes at the character n-gram level, capturing subword overlaps between spelling variants — specifically designed for unstandardized Devanagari text.

2. **Contrastive dense retriever fine-tuning**: Fine-tuning MuRIL on synthetically generated Bhojpuri QA pairs using MultipleNegativesRankingLoss, enabling semantically-aware retrieval in a zero-resource setting.

3. **Hybrid RRF fusion**: Combining orthography-robust sparse retrieval with semantics-aware dense retrieval via Reciprocal Rank Fusion, achieving complementary coverage.

4. **Hindi-bridge prompting**: A novel RAG prompting strategy that leverages Hindi as a reasoning bridge for Bhojpuri, exploiting linguistic proximity to improve generation quality.

5. **Reproducible low-resource NLP research framework**: A modular, config-driven codebase with comprehensive evaluation, ablation studies, error analysis, and paper-ready outputs.

## Baselines

| System           | Description                                                    |
| ---------------- | -------------------------------------------------------------- |
| Word BM25        | Standard word-level BM25 (Okapi) with whitespace tokenization  |
| Char N-gram BM25 | Character n-gram BM25 with configurable n-gram range (2-4)     |
| Zero-shot MuRIL  | MuRIL-base-cased without fine-tuning, FAISS retrieval          |
| Fine-tuned MuRIL | MuRIL fine-tuned on synthetic Bhojpuri QA pairs                |
| Hybrid (RRF)     | Char n-gram BM25 + Fine-tuned MuRIL via Reciprocal Rank Fusion |

## Evaluation Metrics

- **MRR@10** — Mean Reciprocal Rank at 10
- **Recall@5** — Fraction of relevant documents in top-5
- **NDCG@10** — Normalized Discounted Cumulative Gain at 10
- **Precision@5** — Fraction of top-5 results that are relevant
- **MAP** — Mean Average Precision

## Results

### Retrieval Comparison
![Retrieval Comparison](paper_assets/figures/retrieval_comparison.png)

### Ablation Study
![Ablation Heatmap](paper_assets/figures/ablation_heatmap.png)

## Project Structure

```
BhojRAG/
├── configs/default.yaml           # Experiment configuration
├── data/
│   ├── raw/sample_corpus.txt      # Sample Bhojpuri corpus
│   ├── processed/                 # Cleaned, chunked data
│   └── synthetic/                 # Generated QA pairs
├── models/                        # Saved model checkpoints
├── src/
│   ├── data/                      # Ingestion, preprocessing, chunking, QA generation
│   ├── retrieval/                 # Sparse, dense, hybrid retrievers
│   ├── rag/                       # Generation pipeline, prompts, LLM backends
│   ├── eval/                      # Metrics, evaluation, error analysis, plotting
│   └── utils/                     # Config, logging, I/O, seed control
├── scripts/                       # Numbered experiment scripts
├── notebooks/                     # Exploration notebooks
├── tests/                         # Unit tests
├── paper_assets/                  # Tables, figures, diagrams
├── outputs/                       # Experiment results
├── README.md
├── requirements.txt
└── .gitignore
```

## Quick Start

### 1. Setup

```bash
git clone https://github.com/im-anishraj/BhojRAG.git
cd BhojRAG
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
python scripts/01_prepare_data.py --config configs/default.yaml
```

### 3. Generate Synthetic QA Pairs

```bash
python scripts/02_generate_qa.py --config configs/default.yaml
```

### 4. Fine-tune Dense Retriever

```bash
python scripts/03_train_dense.py --config configs/default.yaml
```

### 5. Build Retrieval Indices

```bash
python scripts/04_build_indices.py --config configs/default.yaml
```

### 6. Evaluate All Retrievers

```bash
python scripts/05_evaluate.py --config configs/default.yaml
```

### 7. Run RAG Generation

```bash
python scripts/06_run_rag.py --config configs/default.yaml --query "भोजपुरी के इतिहास का ह?"
```

### 8. Generate Paper Assets

```bash
python scripts/07_generate_paper_assets.py --config configs/default.yaml
```

## Configuration

All experiments are driven by YAML config files in `configs/`. The default config (`configs/default.yaml`) provides sensible defaults. Override for specific experiments:

```bash
python scripts/05_evaluate.py --config configs/ablation_ngram.yaml
```

Key config sections: `data`, `sparse`, `dense`, `training`, `hybrid`, `generation`, `evaluation`.

## Experiment Tracking

- **MLflow** (default): Results tracked in `mlruns/`. View with `mlflow ui`.
- **JSON fallback**: Every run saves a JSON record in `outputs/`.

## Target Venues

- ACL / EMNLP / NAACL (Findings track)
- LREC-COLING
- AACL-IJCNLP
- Workshop: AfricaNLP / LowResourceNLP / VarDial

## License

MIT License

## Citation

```bibtex
@misc{bhojrag2026,
  title={BhojRAG: Robust Retrieval-Augmented Generation for Unstandardized Low-Resource Indic Languages},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/yourusername/BhojRAG}}
}
```
