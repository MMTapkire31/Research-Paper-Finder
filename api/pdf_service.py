import fitz  # PyMuPDF
import re
from typing import Optional


def extract_text_from_pdf(file_bytes: bytes) -> dict:
    """
    Extract full text, metadata, and structure from a PDF using PyMuPDF.
    Returns: title, authors, abstract, full_text, sections, page_count, metadata
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        return {'error': f'Could not open PDF: {str(e)}'}

    result = {
        'page_count': doc.page_count,
        'metadata': {},
        'title': '',
        'authors': '',
        'abstract': '',
        'full_text': '',
        'sections': [],
    }

    # Extract metadata
    meta = doc.metadata
    if meta:
        result['metadata'] = {
            'title': meta.get('title', ''),
            'author': meta.get('author', ''),
            'subject': meta.get('subject', ''),
            'creator': meta.get('creator', ''),
        }

    # Extract all text page by page
    all_text = []
    for page_num in range(min(doc.page_count, 30)):  # cap at 30 pages
        page = doc[page_num]
        text = page.get_text("text")
        all_text.append(text)

    full_text = '\n'.join(all_text)
    result['full_text'] = full_text[:15000]  # cap for NLP

    # Parse title (first large bold text or first non-empty line on page 1)
    if meta.get('title'):
        result['title'] = meta['title']
    else:
        first_page_text = all_text[0] if all_text else ''
        lines = [l.strip() for l in first_page_text.split('\n') if l.strip()]
        result['title'] = lines[0] if lines else 'Unknown Title'

    # Parse authors (from metadata or heuristic: line after title)
    if meta.get('author'):
        result['authors'] = meta['author']
    else:
        first_page_text = all_text[0] if all_text else ''
        lines = [l.strip() for l in first_page_text.split('\n') if l.strip()]
        if len(lines) > 1:
            result['authors'] = lines[1]

    # Extract abstract
    abstract = _extract_abstract(full_text)
    result['abstract'] = abstract

    # Extract sections
    sections = _extract_sections(full_text)
    result['sections'] = sections

    doc.close()
    return result


def _extract_abstract(text: str) -> str:
    """Extract abstract section from paper text."""
    patterns = [
        r'(?i)abstract[\s\n:—-]+(.*?)(?=\n\s*(?:1\.?\s+introduction|keywords|index terms|\n\n))',
        r'(?i)abstract[\s\n]+([\s\S]{100,1500})(?=\n\s*\d+\.?\s+\w|\nkeyword)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            abstract = re.sub(r'\s+', ' ', abstract)
            if len(abstract) > 50:
                return abstract[:2000]

    # Fallback: take first 500 chars after first page header noise
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
    if paragraphs:
        return paragraphs[0][:1000]
    return text[:500]


def _extract_sections(text: str) -> list[dict]:
    """Detect section headers and extract content."""
    section_pattern = re.compile(
        r'\n\s*(\d+\.?\s+[A-Z][A-Za-z\s]+|[A-Z][A-Z\s]{3,})\s*\n',
    )
    matches = list(section_pattern.finditer(text))
    sections = []

    for i, match in enumerate(matches[:12]):  # max 12 sections
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else start + 2000
        content = text[start:end].strip()
        content = re.sub(r'\s+', ' ', content)
        if len(content) > 50:
            sections.append({
                'title': title,
                'content': content[:1000],
            })

    return sections


def extract_references(text: str) -> list[str]:
    """Extract reference list from paper."""
    ref_pattern = re.compile(
        r'(?i)(?:references|bibliography)\s*\n([\s\S]+?)(?:\Z)',
        re.DOTALL
    )
    match = ref_pattern.search(text)
    if not match:
        return []

    ref_text = match.group(1)
    # Split on numbered references like [1], [2] or 1. 2.
    refs = re.split(r'\n\s*(?:\[\d+\]|\d+\.)\s+', ref_text)
    refs = [r.strip().replace('\n', ' ') for r in refs if len(r.strip()) > 20]
    return refs[:30]
