#!/usr/bin/env python3
"""
Train a mortality prediction model for bacteremia episodes.

Example:
    python src/models/train_mortality_model.py \
        --input-path data/merged_bacthecom.csv \
        --model-path models/mortality_model.joblib \
        --report-path results/mortality_model_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TRUE_VALUES = {"si", "sí", "yes", "true", "1", "positivo", "positive"}
FALSE_VALUES = {"no", "false", "0", "negativo", "negative"}
TARGET_COLUMN = "mortalidad"

NUMERIC_FEATURES = [
    "edad",
    "duracion_UCI",
    "indice_de_charlson",
    "escala_karnofsky",
    "barthel_inf_90",
    "qsofa",
    "temperatura",
    "frec_cardiaca",
    "frecuencia_respiratoria",
    "tension_arterial_sist",
    "tension_arterial_diast",
    "saturacion_pO2",
    "dias_hasta_cultivo",
    "estancia_total",
    # comorbidity indicators
    "infarto",
    "insuficiencia_cardiaca",
    "evp",
    "e_cerebrovascular",
    "demencia",
    "e_pulmonar_cronica",
    "ulcera_peptica",
    "colagenopatia",
    "hemiplejia",
    "erc",
    "neoplasia_tratamiento_activo",
    "neoplasia_solida_metastasica",
    "neoplasia_solida_no_metastasica",
    "linfoma",
    "leucemia",
    "sida",
    "hepatopatia_ligera",
    "hepatopatia_moderada_o_grave",
    "diabetes",
    "diabetes_sin_lesion_organo_diana",
    "diabetes_con_lesion_organo_diana",
    "inmunosupresion",
    "hospit_ano_previo",
    "hospit_mes_previo",
    "hospit_ano_previo_uci",
]

CATEGORICAL_FEATURES = [
    "sexo",
    "area_hosp",
    "organo_aparato",
    "microorganismo",
    "fenotipo_resistencia",
    "foco_controlable",
    "IRAs_nosocomial",
    "uci_por_el_episodio",
    "sepsis",
    "shock_septico",
    "situacion_funcional_basal",
    "foco",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a mortality prediction model for bacteremia episodes."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/merged_bacthecom.csv"),
        help="Path to the merged dataset CSV.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/mortality_model.joblib"),
        help="Where to store the fitted model pipeline.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("results/mortality_model_report.json"),
        help="Where to store metrics and configuration summary.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for hold-out evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def ensure_columns_present(df: pd.DataFrame, columns: List[str]) -> List[str]:
    return [col for col in columns if col in df.columns]


def coerce_binary(series: pd.Series) -> pd.Series:
    def _convert(value):
        if pd.isna(value):
            return np.nan
        text = str(value).strip().lower()
        if text in TRUE_VALUES:
            return 1.0
        if text in FALSE_VALUES:
            return 0.0
        try:
            return float(value)
        except ValueError:
            return np.nan

    return series.apply(_convert)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = {
        "fecha_ingreso": "fecha_ingreso",
        "fecha_cultivo": "fecha_cultivo",
        "fecha_alta": "fecha_alta",
    }
    for col in date_cols.values():
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if {"fecha_ingreso", "fecha_cultivo"}.issubset(df.columns):
        df["dias_hasta_cultivo"] = (
            df["fecha_cultivo"] - df["fecha_ingreso"]
        ).dt.days.astype("float64")
    else:
        df["dias_hasta_cultivo"] = np.nan

    if {"fecha_ingreso", "fecha_alta"}.issubset(df.columns):
        df["estancia_total"] = (
            df["fecha_alta"] - df["fecha_ingreso"]
        ).dt.days.astype("float64")
    else:
        df["estancia_total"] = np.nan

    binary_cols = [
        col
        for col in [
            "IRAs_nosocomial",
            "uci_por_el_episodio",
            "sepsis",
            "shock_septico",
            "foco_controlable",
        ]
        if col in df.columns
    ]
    for col in binary_cols:
        df[col] = coerce_binary(df[col])

    severity_cols = ["qsofa", "indice_de_charlson", "escala_karnofsky"]
    for col in severity_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_pipeline(
    numeric_features: List[str], categorical_features: List[str]
) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=500,
        solver="lbfgs",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_and_evaluate(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    df = df.copy()
    df = engineer_features(df)

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN])

    numeric_features = ensure_columns_present(df, NUMERIC_FEATURES)
    categorical_features = ensure_columns_present(df, CATEGORICAL_FEATURES)

    feature_columns = numeric_features + categorical_features
    if not feature_columns:
        raise ValueError("No usable features found in the dataset.")

    X = df[feature_columns]
    y = df[TARGET_COLUMN].astype(int)

    model = build_pipeline(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    scoring = ["roc_auc", "average_precision", "accuracy", "precision", "recall"]
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        scoring=scoring,
        cv=5,
        n_jobs=-1,
        return_train_score=False,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "holdout": {
            "roc_auc": roc_auc_score(y_test, y_proba),
            "average_precision": average_precision_score(y_test, y_proba),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
        },
        "cross_validation": {
            metric: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
            for metric, values in cv_results.items()
            if metric.startswith("test_")
        },
        "class_distribution": {
            "positive": int(y.sum()),
            "negative": int(len(y) - y.sum()),
        },
    }

    metrics["classification_report"] = classification_report(
        y_test, y_pred, zero_division=0, output_dict=True
    )

    metadata = {
        "target_column": TARGET_COLUMN,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "n_samples": len(df),
        "n_features": len(feature_columns),
    }

    return {
        "metrics": metrics,
        "metadata": metadata,
        "model": model,
    }


def main():
    args = parse_args()

    df = pd.read_csv(args.input_path)
    results = train_and_evaluate(df, args)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(results["model"], args.model_path)

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    with args.report_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {"metrics": results["metrics"], "metadata": results["metadata"]},
            fp,
            indent=2,
        )

    print("Model saved to:", args.model_path)
    print("Report saved to:", args.report_path)
    print(
        json.dumps(
            results["metrics"]["holdout"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

