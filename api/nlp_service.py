import re
import math
from functools import lru_cache
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class SemanticSearchUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _get_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SemanticSearchUnavailable(
            "Semantic search dependencies are missing. Install requirements.txt."
        ) from exc
    return SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        revision=EMBEDDING_MODEL_REVISION,
    )


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\-]', ' ', text)
    return text.strip().lower()


def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    """Extract top keywords from text using TF-IDF on a single document."""
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        sentences = [text]

    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=200,
            min_df=1
        )
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = scores.argsort()[::-1][:top_n]
        keywords = [feature_names[i] for i in top_indices if len(feature_names[i]) > 3]
        return keywords[:top_n]
    except Exception:
        words = clean_text(text).split()
        freq = Counter(words)
        stopwords = {'the','a','an','and','or','but','in','on','at','to','for',
                     'of','with','by','from','is','are','was','were','be','been',
                     'have','has','had','do','does','did','will','would','could',
                     'should','may','might','this','that','these','those','it',
                     'its','their','they','them','we','our','you','your','he','she'}
        keywords = [w for w, _ in freq.most_common(50) if w not in stopwords and len(w) > 3]
        return keywords[:top_n]


def rank_papers_tfidf(query: str, papers: list[dict]) -> list[dict]:
    """Rank papers by TF-IDF cosine similarity against query."""
    if not papers:
        return papers

    documents = []
    for p in papers:
        doc = f"{p.get('title', '')} {p.get('title', '')} {p.get('abstract', '')} {' '.join(p.get('categories', []))}"
        documents.append(doc)

    corpus = [query] + documents
    try:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vec = tfidf_matrix[0]
        doc_vecs = tfidf_matrix[1:]
        similarities = cosine_similarity(query_vec, doc_vecs).flatten()

        for i, paper in enumerate(papers):
            paper['relevance_score'] = round(float(similarities[i]), 4)

        papers.sort(key=lambda x: x['relevance_score'], reverse=True)
    except Exception:
        for p in papers:
            p['relevance_score'] = 0.0

    return papers


def rank_papers_semantic(query: str, papers: list[dict]) -> list[dict]:
    """Rank papers by cosine similarity between sentence-transformer embeddings."""
    if not papers:
        return papers

    documents = [
        f"{paper.get('title', '')} {paper.get('abstract', '')}".strip()
        for paper in papers
    ]
    indexed_positions = [i for i, document in enumerate(documents) if document]
    for paper in papers:
        paper['relevance_score'] = 0.0
    if not query.strip() or not indexed_positions:
        return papers

    try:
        import faiss
    except ImportError as exc:
        raise SemanticSearchUnavailable(
            "Semantic search dependencies are missing. Install requirements.txt."
        ) from exc

    model = _get_embedding_model()
    vectors = model.encode(
        [documents[i] for i in indexed_positions],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    scores, result_positions = index.search(query_vector, len(indexed_positions))
    for score, result_position in zip(scores[0], result_positions[0]):
        paper_position = indexed_positions[int(result_position)]
        papers[paper_position]['relevance_score'] = round(float(score), 4)

    papers.sort(key=lambda paper: paper['relevance_score'], reverse=True)
    return papers


def summarize_abstract(abstract: str) -> dict:
    """Extract structured info from abstract using heuristics + TF-IDF."""
    keywords = extract_keywords(abstract, top_n=8)

    sentences = re.split(r'(?<=[.!?])\s+', abstract.strip())
    sentences = [s for s in sentences if len(s) > 20]

    contribution = sentences[0] if sentences else abstract[:200]
    conclusion = sentences[-1] if len(sentences) > 1 else ''

    method_keywords = ['propose', 'present', 'introduce', 'develop', 'design',
                       'model', 'method', 'approach', 'framework', 'algorithm',
                       'network', 'system', 'architecture']
    result_keywords = ['achieve', 'outperform', 'improve', 'show', 'demonstrate',
                       'result', 'accuracy', 'performance', 'state-of-the-art', 'sota']

    method_sentences = [s for s in sentences if any(kw in s.lower() for kw in method_keywords)]
    result_sentences = [s for s in sentences if any(kw in s.lower() for kw in result_keywords)]

    return {
        'keywords': keywords,
        'contribution': contribution,
        'method': method_sentences[0] if method_sentences else '',
        'results': result_sentences[0] if result_sentences else conclusion,
        'word_count': len(abstract.split()),
    }


def compute_similarity_matrix(papers: list[dict]) -> list[dict]:
    """Find similar papers among a list using TF-IDF pairwise similarity."""
    if len(papers) < 2:
        return papers

    documents = [f"{p.get('title','')} {p.get('abstract','')}" for p in papers]
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
        tfidf_matrix = vectorizer.fit_transform(documents)
        sim_matrix = cosine_similarity(tfidf_matrix)

        for i, paper in enumerate(papers):
            sims = [(j, sim_matrix[i][j]) for j in range(len(papers)) if j != i]
            sims.sort(key=lambda x: x[1], reverse=True)
            paper['similar_indices'] = [s[0] for s in sims[:3]]
    except Exception:
        pass

    return papers
