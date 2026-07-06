<div align="center">

# <u>🧠 MKGF — Multimodal Knowledge-Guided Fusion for Medical VQA</u>

**A Retrieval-Augmented Generation pipeline for Visual Question Answering on brain MRI scans**

Built during a summer research internship at **IIIT Bhagalpur**, Dept. of CSE

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.37.2-FFD21E?style=flat-square)
![Status](https://img.shields.io/badge/status-research%20prototype-orange?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

[Overview](#-overview) • [Architecture](#-architecture) • [Results](#-results) • [Setup](#️-setup) • [Repo Structure](#-repo-structure) • [Limitations](#-what-this-is--and-isnt)

</div>

---

## <u>📋 Overview</u>

Given a brain **MRI image** and a free-text **clinical question**, MKGF:

1. 🔎 **Retrieves** relevant records from a structured medical knowledge base
2. 🎯 **Reranks** them using true multimodal (text + image) similarity
3. 💬 **Generates** a grounded answer using a medical vision-language model

Think of it as *"Google search for a radiologist"* — instead of asking a language model to hallucinate a diagnosis from scratch, we first pull the most relevant prior cases and hand them to the model as grounding context before it answers.

📄 **Full technical writeup:** [`docs/IIIT_Bhagalpur_Internship_Report_final_.pdf`](docs/IIIT_Bhagalpur_Internship_Report_final_.pdf)

---

## <u>🏗️ Architecture</u>

```
   🖼️ + ❓                    ┌─────────────────────┐
  MRI Image  ──────────────▶ │   BiomedCLIP +        │
  + Question                 │   ColBERT Retriever   │──┐
                              └─────────────────────┘  │  top-8
                                                         ▼
                              ┌─────────────────────┐
                              │  Multimodal Reranking │
                              │  (text + image sim)   │
                              └─────────────────────┘
                                                         │ top-3
                                                         ▼
                              ┌─────────────────────┐
                              │     LLaVA-Med          │
                              │   (Mistral-7B)         │──────▶  ✅ Answer
                              │  grounded generation    │
                              └─────────────────────┘
                                        ▲
                                        │
                          📚 Structured Knowledge Base
                          (92 annotated MRI records)
```

| Stage | What happens |
|---|---|
| **1. Query Encoding** | Question tokenized (PubMedBERT tokenizer) → encoded via a frozen BiomedCLIP text tower + trained projection head |
| **2. Coarse Retrieval** | Cosine similarity vs. a pre-computed knowledge-base embedding matrix → top-8 candidates |
| **3. Multimodal Reranking** | Candidates reranked using combined text + image similarity against the *actual* query MRI → top-3 survive |
| **4. Grounded Generation** | Top-3 chunks formatted into a structured prompt, passed with the image to **LLaVA-Med (Mistral-7B)** |

---

## <u>📊 Results</u>

*Evaluated on N = 92 held-out knowledge-base entries*

<div align="center">

| 🎯 Retrieval Accuracy | 💬 Generation Accuracy | 📈 F1 Score |
|:---:|:---:|:---:|
| **80.43%** | **65.22%** | **83.85%** |

</div>

<details>
<summary><strong>📐 Full metrics breakdown</strong> (click to expand)</summary>

| Metric | Score |
|---|---|
| Exact Match | 65.22% |
| F1 Score | 83.85% |
| Precision | 89.51% |
| Recall | 81.57% |
| BLEU-1 | 78.09% |
| BLEU-2 | 73.00% |
| BLEU-3 | 70.68% |
| BLEU-4 | 70.60% |
| Accuracy (substring match) | 65.22% |

📁 Raw numbers: [`results/evaluation_metrics.json`](results/evaluation_metrics.json) · 📝 Discussion: [`docs/results.md`](docs/results.md)

</details>

---

## <u>⚠️ What this is — and isn't</u>

> Honesty about limitations is part of the engineering, not a footnote.

- **📛 "Knowledge graph" naming caveat** — the knowledge base (`merged_u.json`, 92 records) is architecturally a **flat JSON lookup table**, not a knowledge graph in the formal sense. There are no explicit typed edges or multi-hop traversal anywhere in the pipeline. More accurately: a *structured medical knowledge base used as a retrieval corpus*.
- **⚖️ Class imbalance** — 49 of 92 entities (~53%) are labelled Glioblastoma Multiforme / High-Grade Glioma. Headline accuracy numbers should be read with this in mind.
- **📉 Retrieval–generation gap** — retrieval accuracy (80.43%) noticeably exceeds generation accuracy (65.22%), suggesting part of the bottleneck sits in the generation/answer-formatting stage rather than retrieval quality alone.

Full discussion in Section 8 ("Critical Analysis of Results") of the report.

---

## <u>🗂️ Repo Structure</u>

```
mkgf-medical-vqa/
├── 📓 notebooks/
│   └── mkgf_pipeline.ipynb        # full end-to-end Kaggle notebook
├── 🧩 src/
│   ├── retriever/
│   │   ├── colbert_model.py       # ColBERT retriever, NTXentLoss
│   │   └── dataset.py             # MedicalReportDataset (synthetic Q&A pairs)
│   ├── patches/
│   │   └── open_clip_patches.py   # torch / open_clip compatibility patches
│   ├── generation/
│   │   └── inference.py           # retrieve → rerank → generate helpers
│   └── evaluation/
│       └── metrics.py             # Exact Match / F1 / Precision / Recall / BLEU
├── 📚 data/
│   └── README.md                  # dataset schema + class-imbalance note
├── 📄 docs/
│   ├── IIIT_Bhagalpur_Internship_Report_final_.pdf
│   └── results.md
├── 📊 results/
│   └── evaluation_metrics.json
├── 💬 examples/
│   └── sample_qa_outputs.md
├── requirements.txt
└── .gitignore
```

---

## <u>⚙️ Setup</u>

> Developed and run entirely on **Kaggle Notebooks** (single NVIDIA Tesla T4, 15.6 GB VRAM)

> 🔁 **Kaggle quirk:** *"Restart Session" wipes all pip installs* — always re-run the install cell first after any restart.

```bash
# 1️⃣ Core extras (Kaggle pre-installs torch/transformers)
pip install transformers==4.37.2 accelerate==0.26.0 bitsandbytes==0.41.3 \
            open_clip_torch==2.20.0 peft einops nltk tabulate

# 2️⃣ LLaVA-Med (not on PyPI — install from source)
git clone https://github.com/microsoft/LLaVA-Med
cd LLaVA-Med && pip install -e .

# 3️⃣ Apply compatibility patches BEFORE importing open_clip anywhere else
python -c "from src.patches.open_clip_patches import apply_all_patches; apply_all_patches()"
```

Then open `notebooks/mkgf_pipeline.ipynb` and run cells top to bottom.

See **Section 5** of the report for the full list of engineering issues (dependency conflicts, tokenizer mismatches, checkpoint-loading incompatibilities) and how each was resolved.

---

## <u>🧰 Tech Stack</u>

<div align="center">

| Component | Model / Library |
|---|---|
| 🔍 Retriever backbone | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` *(frozen)* |
| 🎛️ Retriever heads | Custom ColBERT-style dual text/image projection heads (~1.05M trainable params) |
| 🗣️ Generator | `microsoft/llava-med-v1.5-mistral-7b` *(fp16)* |
| 🛠️ Frameworks | PyTorch 2.10 (cu128) · Transformers 4.37.2 · open_clip_torch 2.20.0 · PEFT · Accelerate · bitsandbytes |

</div>

---

## <u>👥 Team</u>

| Name | Role |
|---|---|
| **Kunal Gaurav** | Retrieval pipeline, retriever training, RAG inference, evaluation |
| **Priyanshu Chakraborty** | Co-contributor, LLM Report Generation + Datasets |
| **Dr. Mehul, Dr. Suneel** | Guidance (IIIT Bhagalpur, CSE Department) |

---

## <u>📜 License</u>

Licensed under **MIT** — see [`LICENSE`](LICENSE).

> ⚠️ This license covers the pipeline code only. Third-party models (BiomedCLIP, LLaVA-Med) and datasets are subject to their own respective licenses — check those before redistribution.

---

||||||| WHOLE REPORT AND DATASETS ARE HIDDEN FOR FUTURE WORKINGS ||||||



<div align="center">

⭐ *If this project is useful to you, consider giving it a star!*

</div>
