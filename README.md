# AI Research Paper Finder

Open browser at http://localhost:8000

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python manage.py migrate
python manage.py runserver
```

Open: http://localhost:8000

Search results are ranked by semantic similarity using the
`sentence-transformers/all-MiniLM-L6-v2` model and a FAISS index. The first
search downloads the pinned model revision; later searches reuse it from the
local cache.

## Search evaluation

Run the reproducible mini-benchmark from the project directory:

```bash
python benchmarks/benchmark_search.py
```

The script downloads the pinned SciFact dataset archive and verifies its
SHA-256 before evaluation. It samples 30 labeled development claims using
seed 42 and ranks the full corpus (5,183 papers with abstracts). Evidence
documents annotated for each claim are treated as relevant. Recall@5 and MRR
are macro-averaged over queries; hybrid ranking uses reciprocal rank fusion
(RRF, k=60). This evaluates scientific evidence retrieval, not general user
search preferences.

Measured on the 30-query sample (38 annotated query-paper pairs):

| Method | Recall@5 | MRR |
| --- | ---: | ---: |
| TF-IDF | 0.407 | 0.422 |
| Embeddings | 0.873 | 0.759 |
| Hybrid (RRF) | 0.682 | 0.519 |

Dataset: [SciFact](https://github.com/allenai/scifact), introduced in
[Fact or Fiction: Verifying Scientific Claims](https://arxiv.org/abs/2004.14974).

<img width="1914" height="960" alt="image" src="https://github.com/user-attachments/assets/43ece0b3-d07d-4539-afcb-cf0a9c7af6ad" />
<img width="1911" height="953" alt="image" src="https://github.com/user-attachments/assets/fa54dc86-0d28-47b0-9387-5e5004bf620b" />


├── manage.py
└── requirements.txt
