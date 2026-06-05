"""
model_xgboost.py — Entrenamiento XGBoost + Generación de Señales
=================================================================
1. Carga _all_features.csv
2. Construye X (features) e Y (target = retorno futuro > 0)
3. Divide en Train / Test (80/20 cronológico, sin shuffle)
4. Entrena XGBoostClassifier
5. Genera scores de probabilidad por ticker por día
6. Clasifica señales: BUY / HOLD / SELL
7. Guarda predictions.csv y el modelo entrenado
"""

import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_FEATURES, DATA_SIGNALS, MODELS_DIR,
    TICKERS, BENCHMARK,
    XGBOOST_PARAMS, TRAIN_TEST_SPLIT, FORWARD_RETURN_DAYS,
    KAXANUK_FEATURES,
)
from features import get_feature_columns

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_PATH      = MODELS_DIR / "xgboost_model.pkl"
PREDICTIONS_OUT = DATA_SIGNALS / "predictions.csv"

# Umbral de probabilidad para señales
BUY_THRESHOLD  = 0.55   # prob > 55% → BUY
SELL_THRESHOLD = 0.45   # prob < 45% → SELL


def load_features() -> pd.DataFrame:
    path = DATA_FEATURES / "_all_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"No encontrado: {path}. Ejecuta features.py primero.")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    log.info(f"Features cargadas: {df.shape[0]:,} filas | {df.shape[1]} columnas")
    return df


def build_dataset(df: pd.DataFrame):
    """
    Separa X (features) e Y (target).
    Elimina filas con NaN en features o target.
    """
    feature_cols = get_feature_columns(df)

    # Usar todos los tickers excepto el benchmark para entrenar
    if "Ticker" in df.columns:
        df_train_universe = df[df["Ticker"] != BENCHMARK].copy()
    else:
        df_train_universe = df.copy()

    # Solo columnas que realmente existen
    valid_feature_cols = [c for c in feature_cols if c in df_train_universe.columns]
    feature_cols = valid_feature_cols

    df_clean = df_train_universe.dropna(subset=feature_cols + ["target"])

    X = df_clean[feature_cols]
    y = df_clean["target"]

    log.info(f"Dataset: {X.shape[0]:,} filas | {X.shape[1]} features")
    log.info(f"Balance target → BUY: {y.mean():.1%} | SELL/HOLD: {1-y.mean():.1%}")

    return X, y, df_clean, feature_cols


def train_test_split_temporal(X: pd.DataFrame, y: pd.Series, split: float):
    """
    Split cronológico (sin shuffle) para evitar data leakage.
    """
    n = len(X)
    cutoff = int(n * split)
    return X.iloc[:cutoff], X.iloc[cutoff:], y.iloc[:cutoff], y.iloc[cutoff:]


def train_model(X_train, y_train) -> XGBClassifier:
    log.info("Entrenando XGBoost...")
    model = XGBClassifier(**XGBOOST_PARAMS, eval_metric="logloss")
    model.fit(X_train, y_train)
    log.info("  ✓ Modelo entrenado")
    return model


def evaluate_model(model: XGBClassifier, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    log.info(f"\n{'='*50}")
    log.info(f"  AUC-ROC: {auc:.4f}")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=['SELL/HOLD','BUY'])}")
    log.info(f"{'='*50}")
    return auc


def generate_signals(model: XGBClassifier, df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Genera señales para TODOS los tickers en TODAS las fechas.
    Incluye benchmark (SPY) para referencia pero no se opera.
    """
    feature_cols = get_feature_columns(df_all)
    df_valid = df_all.dropna(subset=feature_cols)

    scores = model.predict_proba(df_valid[feature_cols])[:, 1]

    signals_df = df_valid[["Ticker", "Close"]].copy()
    signals_df["score"] = scores
    signals_df["signal"] = "HOLD"
    signals_df.loc[signals_df["score"] >= BUY_THRESHOLD,  "signal"] = "BUY"
    signals_df.loc[signals_df["score"] <= SELL_THRESHOLD, "signal"] = "SELL"

    log.info(f"\nDistribución de señales:")
    log.info(signals_df["signal"].value_counts().to_string())

    return signals_df


def run_model() -> pd.DataFrame:
    # 1. Cargar features
    df = load_features()

    # 2. Construir dataset
    X, y, df_clean, feature_cols = build_dataset(df)

    # 3. Split temporal
    X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, TRAIN_TEST_SPLIT)
    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 4. Entrenar modelo
    model = train_model(X_train, y_train)

    # 5. Evaluar
    evaluate_model(model, X_test, y_test)

    # 6. Guardar modelo
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    log.info(f"  💾 Modelo guardado: {MODEL_PATH}")

    # 7. Generar señales sobre todos los datos
    signals = generate_signals(model, df)
    signals.to_csv(PREDICTIONS_OUT)
    log.info(f"  📊 Señales guardadas: {PREDICTIONS_OUT}")

    return signals


if __name__ == "__main__":
    run_model()
