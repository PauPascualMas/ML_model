from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),          # Home page / form
    path("predict/", views.predict, name="predict"),   # Prediction endpoint
    path("explain/", views.explain, name="explain"),   # SHAP explanation
    path("model-info/", views.model_metadata, name="model-info"),  # Model metadata
]
