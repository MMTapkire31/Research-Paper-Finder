"""Evaluate paper retrieval on a fixed, labeled sample from SciFact."""

import argparse
import hashlib
import json
import random
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.nlp_service import (  # noqa: E402
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    _get_embedding_model,
    rank_papers_tfidf,
)


DATASET_URL = (
    "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
    "?versionId=8LiW3OUBLBvhehDzrRdGD1ibIaXL.6PQ"
)
DATASET_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
RRF_K = 60


def _read_jsonl_member(archive, member_name):
    member = archive.extractfile(member_name)
    if member is None:
        raise ValueError(f"Dataset archive is missing {member_name}.")
    return [json.loads(line) for line in member]


def load_scifact(archive_path):
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if digest != DATASET_SHA256:
        raise ValueError("SciFact archive checksum does not match the pinned dataset.")

    with tarfile.open(archive_path, "r:gz") as archive:
        claims = _read_jsonl_member(archive, "data/claims_dev.jsonl")
        corpus = _read_jsonl_member(archive, "data/corpus.jsonl")

    papers = [
        {
            "id": str(document["doc_id"]),
            "title": document.get("title", ""),
            "abstract": document.get("abstract", ""),
        }
        for document in corpus
        if document.get("abstract")
    ]
    paper_ids = {paper["id"] for paper in papers}
    labeled_claims = []
    for claim in claims:
        relevant_ids = set(claim.get("evidence", {})) & paper_ids
        if relevant_ids:
            labeled_claims.append(
                {
                    "id": str(claim["id"]),
                    "query": claim["claim"],
                    "relevant_ids": relevant_ids,
                }
            )
    return papers, labeled_claims


def reciprocal_rank_fusion(rankings, paper_ids, rrf_k=RRF_K):
    scores = {paper_id: 0.0 for paper_id in paper_ids}
    for ranking in rankings:
        for rank, paper_id in enumerate(ranking, start=1):
            scores[paper_id] += 1 / (rrf_k + rank)
    return sorted(paper_ids, key=lambda paper_id: (-scores[paper_id], paper_id))


def calculate_metrics(rankings, relevance, cutoff=5):
    recall_scores = []
    reciprocal_ranks = []
    for query_id, relevant_ids in relevance.items():
        ranking = rankings[query_id]
        top_relevant = set(ranking[:cutoff]) & relevant_ids
        recall_scores.append(len(top_relevant) / len(relevant_ids))

        first_relevant_rank = next(
            (rank for rank, paper_id in enumerate(ranking, start=1) if paper_id in relevant_ids),
            None,
        )
        reciprocal_ranks.append(1 / first_relevant_rank if first_relevant_rank else 0.0)

    if not recall_scores:
        raise ValueError("No labeled queries are available for evaluation.")
    return {
        f"recall@{cutoff}": sum(recall_scores) / len(recall_scores),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }


def _semantic_rankings(queries, papers):
    import faiss

    model = _get_embedding_model()
    documents = [f"{paper['title']} {paper['abstract']}".strip() for paper in papers]
    vectors = model.encode(
        documents,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    query_vectors = model.encode(
        queries,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
    _, positions = index.search(query_vectors, len(papers))
    paper_ids = [paper["id"] for paper in papers]
    return [[paper_ids[position] for position in row] for row in positions]


def _tfidf_rankings(queries, papers):
    rankings = []
    for query in queries:
        ranked = rank_papers_tfidf(query, [paper.copy() for paper in papers])
        rankings.append([paper["id"] for paper in ranked])
    return rankings


def run_benchmark(archive_path, query_count=30, seed=42):
    papers, claims = load_scifact(archive_path)
    if query_count < 1:
        raise ValueError("Query count must be at least 1.")
    if query_count > len(claims):
        raise ValueError(
            f"Requested {query_count} queries, but only {len(claims)} have relevance labels."
        )

    claims.sort(key=lambda claim: int(claim["id"]))
    selected = random.Random(seed).sample(claims, query_count)
    selected.sort(key=lambda claim: int(claim["id"]))
    queries = [claim["query"] for claim in selected]
    relevance = {claim["id"]: claim["relevant_ids"] for claim in selected}

    tfidf_rankings = _tfidf_rankings(queries, papers)
    semantic_rankings = _semantic_rankings(queries, papers)
    paper_ids = [paper["id"] for paper in papers]
    hybrid_rankings = [
        reciprocal_rank_fusion((tfidf, semantic), paper_ids)
        for tfidf, semantic in zip(tfidf_rankings, semantic_rankings)
    ]
    rankings_by_method = {
        "TF-IDF": tfidf_rankings,
        "Embeddings": semantic_rankings,
        "Hybrid (RRF)": hybrid_rankings,
    }
    results = {}
    for method, rankings in rankings_by_method.items():
        query_rankings = {
            claim["id"]: ranking
            for claim, ranking in zip(selected, rankings)
        }
        results[method] = calculate_metrics(query_rankings, relevance)

    return results, len(papers), len(selected), sum(map(len, relevance.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-archive",
        type=Path,
        help="Use a previously downloaded SciFact data.tar.gz archive.",
    )
    args = parser.parse_args()

    if args.data_archive:
        archive_path = args.data_archive
        results, corpus_count, query_count, relevant_count = run_benchmark(
            archive_path, args.queries, args.seed
        )
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "scifact-data.tar.gz"
            urllib.request.urlretrieve(DATASET_URL, archive_path)
            results, corpus_count, query_count, relevant_count = run_benchmark(
                archive_path, args.queries, args.seed
            )

    print(
        f"SciFact dev | {query_count} queries | {corpus_count} papers | "
        f"{relevant_count} query-paper relevance pairs | seed={args.seed}"
    )
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"Embedding model revision: {EMBEDDING_MODEL_REVISION}")
    print("| Method | Recall@5 | MRR |")
    print("| --- | ---: | ---: |")
    for method, metrics in results.items():
        print(
            f"| {method} | {metrics['recall@5']:.3f} "
            f"| {metrics['mrr']:.3f} |"
        )


if __name__ == "__main__":
    main()
