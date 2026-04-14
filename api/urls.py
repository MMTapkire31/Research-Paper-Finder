from django.urls import path
from .views import home, SearchView, AnalyzeView, PDFUploadView, SimilarView

urlpatterns = [
    path('', home, name='home'),
    path('api/search/', SearchView.as_view()),
    path('api/analyze/', AnalyzeView.as_view()),
    path('api/upload/', PDFUploadView.as_view()),
    path('api/similar/', SimilarView.as_view()),
]
