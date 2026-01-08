# Base image
FROM python:3.11-slim

# ------------------------
# Environment variables
# ------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ------------------------
# Working directory
# ------------------------
WORKDIR /app

# ------------------------
# System dependencies
# ------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ------------------------
# Copy the consolidated requirements
# ------------------------
COPY requirements.txt /app/requirements.txt

# ------------------------
# Install Python dependencies
# ------------------------
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# ------------------------
# Copy full project
# ------------------------
COPY . /app/

# ------------------------
# Collect static files (Django)
# ------------------------
RUN python manage.py collectstatic --noinput

# ------------------------
# Expose port
# ------------------------
EXPOSE 8000

# ------------------------
# Run Django app with Gunicorn
# ------------------------
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "4"]
