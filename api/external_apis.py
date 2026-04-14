import requests
import xml.etree.ElementTree as ET
from typing import Optional
import urllib.parse

ARXIV_BASE = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"

ARXIV_NS = 'http://www.w3.org/2005/Atom'
ARXIV_NS_ARXIV = 'http://arxiv.org/schemas/atom'


def search_arxiv(query: str, max_results: int = 10, category: str = '') -> list[dict]:
    """Search arXiv API and return structured paper list."""
    search_query = f"all:{query}"
    if category:
        search_query = f"cat:{category} AND all:{query}"

    params = {
        'search_query': search_query,
        'start': 0,
        'max_results': max_results,
        'sortBy': 'relevance',
        'sortOrder': 'descending',
    }

    try:
        resp = requests.get(ARXIV_BASE, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        return []

    return _parse_arxiv_response(resp.text)


def _parse_arxiv_response(xml_text: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    ns = {'atom': ARXIV_NS, 'arxiv': ARXIV_NS_ARXIV}
    papers = []

    for entry in root.findall('atom:entry', ns):
        paper_id_raw = entry.findtext('atom:id', '', ns)
        arxiv_id = paper_id_raw.split('/abs/')[-1] if '/abs/' in paper_id_raw else paper_id_raw

        title = entry.findtext('atom:title', '', ns).strip().replace('\n', ' ')
        abstract = entry.findtext('atom:summary', '', ns).strip().replace('\n', ' ')
        published = entry.findtext('atom:published', '', ns)[:10]
        updated = entry.findtext('atom:updated', '', ns)[:10]

        authors = [
            a.findtext('atom:name', '', ns)
            for a in entry.findall('atom:author', ns)
        ]

        categories = [
            c.get('term', '')
            for c in entry.findall('atom:category', ns)
        ]

        pdf_link = ''
        html_link = paper_id_raw
        for link in entry.findall('atom:link', ns):
            if link.get('type') == 'application/pdf':
                pdf_link = link.get('href', '')
            if link.get('rel') == 'alternate':
                html_link = link.get('href', paper_id_raw)

        comment = entry.findtext('arxiv:comment', '', ns)
        journal_ref = entry.findtext('arxiv:journal_ref', '', ns)
        doi = entry.findtext('arxiv:doi', '', ns)

        papers.append({
            'id': arxiv_id,
            'source': 'arxiv',
            'title': title,
            'abstract': abstract,
            'authors': authors,
            'published': published,
            'updated': updated,
            'categories': [c for c in categories if c],
            'pdf_url': pdf_link,
            'html_url': html_link,
            'comment': comment,
            'journal_ref': journal_ref,
            'doi': doi,
            'citation_count': None,
        })

    return papers


def search_semantic_scholar(query: str, max_results: int = 10) -> list[dict]:
    """Search Semantic Scholar API."""
    url = f"{SEMANTIC_SCHOLAR_BASE}/paper/search"
    params = {
        'query': query,
        'limit': max_results,
        'fields': 'title,abstract,authors,year,citationCount,externalIds,url,venue,openAccessPdf,fieldsOfStudy',
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    papers = []
    for item in data.get('data', []):
        authors = [a.get('name', '') for a in item.get('authors', [])]
        ext_ids = item.get('externalIds', {})
        arxiv_id = ext_ids.get('ArXiv', '')
        doi = ext_ids.get('DOI', '')

        pdf_url = ''
        oap = item.get('openAccessPdf')
        if oap:
            pdf_url = oap.get('url', '')

        papers.append({
            'id': item.get('paperId', ''),
            'source': 'semantic_scholar',
            'title': item.get('title', ''),
            'abstract': item.get('abstract', '') or '',
            'authors': authors,
            'published': str(item.get('year', '')),
            'updated': '',
            'categories': item.get('fieldsOfStudy', []) or [],
            'pdf_url': pdf_url,
            'html_url': item.get('url', ''),
            'comment': '',
            'journal_ref': item.get('venue', ''),
            'doi': doi,
            'citation_count': item.get('citationCount'),
            'arxiv_id': arxiv_id,
        })

    return papers


def get_paper_details_semantic_scholar(arxiv_id: str) -> Optional[dict]:
    """Enrich a paper with citation count from Semantic Scholar."""
    url = f"{SEMANTIC_SCHOLAR_BASE}/paper/arXiv:{arxiv_id}"
    params = {'fields': 'citationCount,referenceCount,influentialCitationCount,venue,year'}
    try:
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None
