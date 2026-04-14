import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from .external_apis import search_arxiv, search_semantic_scholar
from .nlp_service import rank_papers_tfidf, summarize_abstract, extract_keywords
from .pdf_service import extract_text_from_pdf, extract_references


def home(request):
    return render(request, 'index.html')


@method_decorator(csrf_exempt, name='dispatch')
class SearchView(View):
    def get(self, request):
        query = request.GET.get('q', '').strip()
        if not query:
            return JsonResponse({'error': 'Query parameter q is required.'}, status=400)
        source = request.GET.get('source', 'arxiv')
        max_results = min(int(request.GET.get('max', 10)), 25)
        category = request.GET.get('category', '')
        papers = []
        if source in ('arxiv', 'both'):
            papers.extend(search_arxiv(query, max_results=max_results, category=category))
        if source in ('semantic_scholar', 'both'):
            papers.extend(search_semantic_scholar(query, max_results=max_results))
        if not papers:
            return JsonResponse({'results': [], 'count': 0, 'query': query})
        papers = rank_papers_tfidf(query, papers)
        seen, unique = set(), []
        for p in papers:
            key = p['title'].lower()[:60]
            if key not in seen:
                seen.add(key)
                if p.get('abstract'):
                    p['summary'] = summarize_abstract(p['abstract'])
                unique.append(p)
        return JsonResponse({'results': unique, 'count': len(unique), 'query': query, 'source': source})


@method_decorator(csrf_exempt, name='dispatch')
class AnalyzeView(View):
    def post(self, request):
        data = json.loads(request.body)
        abstract = data.get('abstract', '').strip()
        title = data.get('title', '')
        if not abstract:
            return JsonResponse({'error': 'abstract required'}, status=400)
        keywords = extract_keywords(f"{title} {abstract}", top_n=12)
        summary = summarize_abstract(abstract)
        return JsonResponse({'keywords': keywords, 'summary': summary})


@method_decorator(csrf_exempt, name='dispatch')
class PDFUploadView(View):
    def post(self, request):
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file uploaded.'}, status=400)
        pdf_file = request.FILES['file']
        if not pdf_file.name.lower().endswith('.pdf'):
            return JsonResponse({'error': 'Only PDF files supported.'}, status=400)
        if pdf_file.size > 20 * 1024 * 1024:
            return JsonResponse({'error': 'File too large. Max 20MB.'}, status=400)
        extracted = extract_text_from_pdf(pdf_file.read())
        if 'error' in extracted:
            return JsonResponse({'error': extracted['error']}, status=422)
        abstract = extracted.get('abstract', '')
        full_text = extracted.get('full_text', '')
        analyze_text = abstract if len(abstract) > 100 else full_text[:3000]
        keywords = extract_keywords(analyze_text, top_n=12)
        summary = summarize_abstract(analyze_text) if analyze_text else {}
        references = extract_references(full_text)
        return JsonResponse({
            'filename': pdf_file.name,
            'page_count': extracted['page_count'],
            'title': extracted['title'],
            'authors': extracted['authors'],
            'abstract': abstract,
            'sections': extracted['sections'],
            'keywords': keywords,
            'summary': summary,
            'references': references[:10],
            'word_count': len(full_text.split()),
        })


@method_decorator(csrf_exempt, name='dispatch')
class SimilarView(View):
    def post(self, request):
        data = json.loads(request.body)
        title = data.get('title', '')
        abstract = data.get('abstract', '')
        keywords = data.get('keywords', [])
        combined = f"{title} {abstract}"
        query = ' '.join(keywords[:5]) if keywords else ' '.join(extract_keywords(combined, top_n=5))
        if not query.strip():
            return JsonResponse({'error': 'Not enough content.'}, status=400)
        papers = search_arxiv(query, max_results=8)
        papers = rank_papers_tfidf(combined, papers)
        for p in papers:
            if p.get('abstract'):
                p['summary'] = summarize_abstract(p['abstract'])
        return JsonResponse({'query_used': query, 'results': papers, 'count': len(papers)})
