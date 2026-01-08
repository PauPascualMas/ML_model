from django.shortcuts import render
from django.http import JsonResponse
from backend.model import predict_proba, shap_top_features, model_info
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent  # /app/backend -> /app
FEATURES_PATH = BASE_DIR / "model" / "features.pkl"

# Load features once when the server starts
raw_features: list[str] = joblib.load(FEATURES_PATH)

# Prepare readable labels for template
# Exclude 'edad' and 'sexo' since they have dedicated inputs
features = [
    {
        "name": f,
        "label": f.capitalize()
    }
    for f in raw_features
    if f not in ("edad", "sexo")
]

# ------------------------
# Home page
# ------------------------
def home(request):
    return render(request, "index.html", {"features": features})

# ------------------------
# Prediction endpoint
# ------------------------
def predict(request):
    if request.method == "POST":
        data = request.POST.dict()
        prob = predict_proba(data)

        if prob < 0.05:
            band = "Low"
        elif prob < 0.20:
            band = "Moderate"
        else:
            band = "High"

        return JsonResponse({
            "mortality_risk": prob,
            "risk_band": band
        })

# ------------------------
# SHAP explanation endpoint
# ------------------------
def explain(request):
    if request.method == "POST":
        data = request.POST.dict()
        shap_data = shap_top_features(data)
        return JsonResponse(shap_data)

# ------------------------
# Model metadata endpoint
# ------------------------
def model_metadata(request):
    return JsonResponse(model_info())
