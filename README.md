# My_Project-Multimodal-RAG-
Multimodal Knowledge-Grounded Retrieval-Augmented Generation for Medical Visual Question Answering on Brain MRI
A Retrieval-Augmented Generation (RAG) pipeline for Medical Visual
Question Answering (Med-VQA) on brain MRI scans, built during a summer
internship at IIIT Bhagalpur.

Given a brain MRI image and a free-text clinical question, the system
retrieves relevant entries from a structured knowledge base, reranks
them using true multimodal similarity, and generates a grounded answer
using a medical vision-language model.

Architecture

        ┌──────────────┐        ┌──────────────────┐        ┌───────────────────┐
Query → │ BiomedCLIP +  │  top-k │  Multimodal        │ top-3  │  LLaVA-Med         │ → Answer
Image → │ ColBERT       │ ─────▶ │  Reranking          │ ─────▶ │  (Mistral-7B)      │
        │ (retriever)   │        │  (text + image sim) │        │  grounded generation│
        └──────────────┘        └──────────────────┘        └───────────────────┘
                                          ▲
                                          │
                              Structured knowledge base
                              (92 annotated MRI records)


Query Encoding — the question is tokenized with a PubMedBERT
tokenizer and encoded via a frozen BiomedCLIP text tower + trained
projection head.
Coarse Retrieval — cosine similarity against a pre-computed
knowledge-base embedding matrix returns the top-8 candidates.
Multimodal Reranking — candidates are reranked using combined
text + image similarity against the actual query MRI image; top-3
survive.
Grounded Generation — the top-3 chunks are formatted into a
structured prompt and passed, with the image, to LLaVA-Med
(Mistral-7B) for the final answer.


Full technical writeup: docs/IIIT_Bhagalpur_Internship_Report_final_.pdf

Results (N = 92, held-out evaluation)

MetricValueRetrieval Accuracy80.43%Generation Accuracy65.22%F1 Score83.85%Precision / Recall89.51% / 81.57%BLEU-1 → BLEU-478.09% → 70.60%

Full metrics table and discussion: docs/results.md · raw numbers: results/evaluation_metrics.json

What this is — and isn't

This project is described deliberately conservatively, matching the
internship report:


"Knowledge graph" naming caveat. The knowledge base
(merged_u.json, 92 records) is a flat JSON lookup table, not a
knowledge graph in the formal sense — there are no explicit typed
edges or multi-hop graph traversal anywhere in the pipeline. It's more
accurately a structured medical knowledge base used as a retrieval
corpus.
Class imbalance. 49 of 92 entities (~53%) are labelled
Glioblastoma Multiforme / High-Grade Glioma. Headline accuracy numbers
should be read with this in mind — see docs/results.md
for the full discussion.
Retrieval–generation gap. Retrieval accuracy (80.43%) noticeably
exceeds generation accuracy (65.22%), suggesting some of the
bottleneck is in the generation/answer-formatting stage rather than
retrieval quality.


Repo structure

mkgf-medical-vqa/
├── notebooks/
│   └── mkgf_pipeline.ipynb        # full end-to-end Kaggle notebook
├── src/                           # pipeline logic extracted into modules
│   ├── retriever/
│   │   ├── colbert_model.py       # ColBERT retriever, NTXentLoss
│   │   └── dataset.py             # MedicalReportDataset (synthetic Q&A pairs)
│   ├── patches/
│   │   └── open_clip_patches.py   # torch / open_clip compatibility patches
│   ├── generation/
│   │   └── inference.py           # retrieve → rerank → generate helpers
│   └── evaluation/
│       └── metrics.py             # Exact Match / F1 / Precision / Recall / BLEU
├── data/
│   └── README.md                  # dataset schema + class-imbalance note
├── docs/
│   ├── IIIT_Bhagalpur_Internship_Report_final_.pdf
│   └── results.md
├── results/
│   └── evaluation_metrics.json
├── examples/
│   └── sample_qa_outputs.md
├── requirements.txt
└── .gitignore

Setup

This pipeline was developed and run entirely on Kaggle Notebooks
(single NVIDIA Tesla T4, 15.6 GB VRAM). Kaggle-specific quirk: "Restart
Session" wipes all pip installs, so the install cell must always be
re-run first after any restart.

bash# 1. Core extras (Kaggle pre-installs torch/transformers)
pip install transformers==4.37.2 accelerate==0.26.0 bitsandbytes==0.41.3 \
            open_clip_torch==2.20.0 peft einops nltk tabulate

# 2. LLaVA-Med (not on PyPI — install from source)
git clone https://github.com/microsoft/LLaVA-Med
cd LLaVA-Med && pip install -e .

# 3. Apply compatibility patches BEFORE importing open_clip anywhere else
python -c "from src.patches.open_clip_patches import apply_all_patches; apply_all_patches()"

Then open notebooks/mkgf_pipeline.ipynb and run cells top to bottom.
See docs/IIIT_Bhagalpur_Internship_Report_final_.pdf Section 5 for the
full list of engineering issues (dependency conflicts, tokenizer
mismatches, checkpoint-loading incompatibilities) and how each was
resolved.

Tech stack


Retriever backbone: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 (frozen)
Retriever heads: custom ColBERT-style dual text/image projection heads (~1.05M trainable params)
Generator: microsoft/llava-med-v1.5-mistral-7b (fp16)
Frameworks: PyTorch 2.10 (cu128), Transformers 4.37.2, open_clip_torch 2.20.0, PEFT, Accelerate, bitsandbytes


Team


Kunal Gaurav — retrieval pipeline, retriever training, RAG inference, evaluation
Priyanshu Chakraborty — co-contributor
Guidance: Dr. Mehul, Dr. Suneel (IIIT Bhagalpur, CSE Department)


License

Add a license (MIT/Apache-2.0 recommended for research code) before making the repo public — see LICENSE.
