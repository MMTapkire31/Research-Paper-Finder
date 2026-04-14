# AI Research Paper Finder

Single Django project — no npm, no React. Open browser at http://localhost:8000

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

## Structure
research_finder/
├── api/
│   ├── views.py          # Home page + all API endpoints
│   ├── nlp_service.py    # TF-IDF ranking & keyword extraction
│   ├── pdf_service.py    # PyMuPDF text extraction
│   ├── external_apis.py  # arXiv + Semantic Scholar
│   └── urls.py
├── config/
│   ├── settings.py
│   └── urls.py
├── templates/
│   └── index.html        # Complete frontend (HTML/CSS/JS)
├── manage.py
└── requirements.txt
