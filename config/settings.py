from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = os.environ.get('SECRET_KEY', 'django-insecure-local-dev-only')
DEBUG = not os.environ.get('RENDER')   # Render sets RENDER automatically
ALLOWED_HOSTS = [".onrender.com", "localhost", "127.0.0.1"]

INSTALLED_APPS = [
    'django.contrib.staticfiles',
    'api',
]

MIDDLEWARE = [
    'django.middleware.common.CommonMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [BASE_DIR / 'templates'],
    'APP_DIRS': True,
    'OPTIONS': {'context_processors': ['django.template.context_processors.request']},
}]

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']

FILE_UPLOAD_MAX_MEMORY_SIZE = 20 * 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE = 20 * 1024 * 1024

DATABASES = {
    'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': BASE_DIR / 'db.sqlite3'}
}
