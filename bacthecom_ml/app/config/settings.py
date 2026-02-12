import os
from pathlib import Path

# ------------------------
# Base paths
# ------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# ------------------------
# Security
# ------------------------
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "replace-this-in-production")
DEBUG = True
ALLOWED_HOSTS = ["localhost", "127.0.0.1", "0.0.0.0"]

# ------------------------
# Installed apps
# ------------------------
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "predictor",  # your app
]

# ------------------------
# Middleware
# ------------------------
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# ------------------------
# URLs
# ------------------------
ROOT_URLCONF = "config.urls"

# ------------------------
# Templates
# ------------------------
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "predictor" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# ------------------------
# WSGI
# ------------------------
WSGI_APPLICATION = "config.wsgi.application"

# ------------------------
# Database (optional, can use SQLite for audit logs)
# ------------------------
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# ------------------------
# Password validation (default)
# ------------------------
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# ------------------------
# Internationalization
# ------------------------
LANGUAGE_CODE = "en-us"
TIME_ZONE = "Europe/Madrid"  # change as needed
USE_I18N = True
USE_L10N = True
USE_TZ = True

# ------------------------
# Static files
# ------------------------
STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "predictor" / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

# ------------------------
# Default primary key
# ------------------------
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
