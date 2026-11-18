import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from django.test import TestCase

from benchmarks.benchmark_search import calculate_metrics, reciprocal_rank_fusion
from .nlp_service import SemanticSearchUnavailable, rank_papers_semantic


class SemanticRankingTests(TestCase):
    def test_benchmark_metrics_and_hybrid_fusion(self):
        relevance = {
            'q1': {'a', 'c'},
            'q2': {'d'},
        }
        rankings = {
            'q1': ['b', 'a', 'c'],
            'q2': ['d', 'e', 'f'],
        }

        metrics = calculate_metrics(rankings, relevance)
        fused = reciprocal_rank_fusion(
            (['b', 'a', 'c'], ['a', 'd', 'c']),
            {'a', 'b', 'c', 'd'},
        )

        self.assertAlmostEqual(metrics['recall@5'], 1.0)
        self.assertAlmostEqual(metrics['mrr'], 0.75)
        self.assertEqual(fused[0], 'a')

    def test_ranks_papers_by_embedding_similarity(self):
        class FakeModel:
            def encode(self, texts, **kwargs):
                vectors = {
                    'query': [1.0, 0.0],
                    'Different topic': [0.0, 1.0],
                    'Related topic': [0.8, 0.6],
                }
                return np.array([vectors[text] for text in texts], dtype=np.float32)

        class FakeIndex:
            def __init__(self, dimensions):
                self.vectors = None

            def add(self, vectors):
                self.vectors = vectors

            def search(self, query_vector, count):
                similarities = query_vector @ self.vectors.T
                positions = np.argsort(-similarities[0])[:count]
                return similarities[:, positions], positions.reshape(1, -1)

        papers = [
            {'title': 'Different', 'abstract': 'topic'},
            {'title': 'Related', 'abstract': 'topic'},
        ]
        with (
            patch.dict(sys.modules, {'faiss': SimpleNamespace(IndexFlatIP=FakeIndex)}),
            patch('api.nlp_service._get_embedding_model', return_value=FakeModel()),
        ):
            ranked = rank_papers_semantic('query', papers)

        self.assertEqual([paper['title'] for paper in ranked], ['Related', 'Different'])
        self.assertEqual(ranked[0]['relevance_score'], 0.8)
        self.assertEqual(ranked[1]['relevance_score'], 0.0)

    def test_empty_query_keeps_papers_with_zero_scores(self):
        papers = [{'title': 'Paper', 'abstract': 'Abstract'}]

        ranked = rank_papers_semantic('   ', papers)

        self.assertIs(ranked, papers)
        self.assertEqual(papers[0]['relevance_score'], 0.0)

    @patch('api.views.search_arxiv', return_value=[{'title': 'Paper', 'abstract': 'Text'}])
    @patch(
        'api.views.rank_papers_semantic',
        side_effect=SemanticSearchUnavailable('Install requirements.txt.'),
    )
    def test_search_reports_missing_semantic_dependencies(self, rank_papers, search_arxiv):
        response = self.client.get('/api/search/?q=topic')

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()['error'], 'Install requirements.txt.')
