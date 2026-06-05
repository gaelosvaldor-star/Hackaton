"""
features.py — Feature Engineering
==================================
Toma los CSVs limpios de data_curator.py y agrega features adicionales
combinándolas con las que KaxaNuk ya calculó.

Features generados:
  A. Retornos simples    → return_1d, return_5d, return_21d, return_63d
  B. Log-return          → log_return
  C. Volatilidad         → vol_21d, vol_63d (anualizadas)
  D. RSI propio          → rsi_14 (como validación, KaxaNuk ya lo tiene)
  E. SMA-50              → sma_50, price_vs_sma_50
  F. Momentum (ROC)      → momentum_10d, momentum_21d
  G. Volumen ratio       → vol_avg_20d, vol_ratio
  H. Target (Y)          → future_return_5d (retorno a 5 días adelante)
                           target (1 si sube, 0 si baja)
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    TICKERS, BENCHMARK,
    DATA_FEATURES,
    RETURNS_PERIODS, VOL_WINDOWS, RSI_WINDOW,
    SMA_WINDOW, MOMENTUM_WINDOWS, VOL_AVG_WINDOW,
    FORWARD_RETURN_DAYS,
    KAXANUK_FEATURES,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── A & B. Retornos ───────────────────────────────────────────────
def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    for p in RETURNS_PERIODS:
        df[f"return_{p}d"] = df["Close"].pct_change(p)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


# ── C. Volatilidad realizada ──────────────────────────────────────
def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    for w in VOL_WINDOWS:
        df[f"vol_{w}d"] = df["log_return"].rolling(w).std() * np.sqrt(252)
    return df


# ── D. RSI ────────────────────────────────────────────────────────
def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    df[f"rsi_{RSI_WINDOW}"]  = _rsi(df["Close"], RSI_WINDOW)
    df["rsi_overbought"]     = (df[f"rsi_{RSI_WINDOW}"] > 70).astype(int)
    df["rsi_oversold"]       = (df[f"rsi_{RSI_WINDOW}"] < 30).astype(int)
    return df


# ── E. SMA-50 ─────────────────────────────────────────────────────
def add_sma50(df: pd.DataFrame) -> pd.DataFrame:
    df[f"sma_{SMA_WINDOW}"]          = df["Close"].rolling(SMA_WINDOW).mean()
    df[f"price_vs_sma_{SMA_WINDOW}"] = df["Close"] / df[f"sma_{SMA_WINDOW}"] - 1
    return df


# ── F. Momentum ───────────────────────────────────────────────────
def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    for w in MOMENTUM_WINDOWS:
        df[f"momentum_{w}d"] = df["Close"].pct_change(w)
    return df


# ── G. Volumen ratio ──────────────────────────────────────────────
def add_volume(df: pd.DataFrame) -> pd.DataFrame:
    df["vol_avg_20d"] = df["Volume"].rolling(VOL_AVG_WINDOW).mean()
    df["vol_ratio"]   = df["Volume"] / df["vol_avg_20d"].replace(0, np.nan)
    return df


# ── H. Target (variable a predecir) ──────────────────────────────
def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorno futuro a N días (shift negativo = hacia adelante).
    target = 1 si el precio sube, 0 si baja.
    """
    df[f"future_return_{FORWARD_RETURN_DAYS}d"] = df["Close"].pct_change(FORWARD_RETURN_DAYS).shift(-FORWARD_RETURN_DAYS)
    df["target"] = (df[f"future_return_{FORWARD_RETURN_DAYS}d"] > 0).astype(int)
    return df


# ── Pipeline ──────────────────────────────────────────────────────
FEATURE_PIPELINE = [
    add_returns,
    add_volatility,
    add_rsi,
    add_sma50,
    add_momentum,
    add_volume,
    add_target,
]

# Columnas mínimas para considerar fila válida (warm-up)
WARMUP_COLS = [f"sma_{SMA_WINDOW}", f"rsi_{RSI_WINDOW}", f"vol_{VOL_WINDOWS[-1]}d"]

# Features que entran al modelo (X)
def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Devuelve lista de columnas que se usan como features del modelo."""
    exclude = {"Open", "High", "Low", "Close", "Volume", "Ticker",
               f"future_return_{FORWARD_RETURN_DAYS}d", "target"}
    kaxanuk_available = [c for c in KAXANUK_FEATURES if c in df.columns]
    own_features = [c for c in df.columns if c not in exclude
                    and not c.startswith("c_")   # evitar duplicar KaxaNuk
                    and c != "target"
                    and "future" not in c]
    return own_features + kaxanuk_available


def engineer_ticker(ticker: str) -> pd.DataFrame | None:
    path = DATA_FEATURES / f"{ticker}.csv"
    if not path.exists():
        log.warning(f"  ✗ {ticker}: archivo curado no encontrado")
        return None

    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df = df.drop(columns=["Ticker"], errors="ignore")

    for step in FEATURE_PIPELINE:
        df = step(df)

    # Eliminar periodo de warm-up
    df = df.dropna(subset=WARMUP_COLS)

    df["Ticker"] = ticker
    log.info(f"  ✓ {ticker}: {len(df):,} filas | features: {len(get_feature_columns(df))}")
    return df


def engineer_features() -> pd.DataFrame:
    """
    Aplica feature engineering a todos los tickers.
    Genera:
      - data/features/<TICKER>_features.csv   → features por acción
      - data/features/_all_features.csv       → consolidado
    """
    all_tickers = TICKERS + ([BENCHMARK] if BENCHMARK not in TICKERS else [])
    all_frames  = []
    failed      = []

    log.info(f"Generando features para {len(all_tickers)} tickers...")

    for ticker in all_tickers:
        df = engineer_ticker(ticker)
        if df is not None:
            df.to_csv(DATA_FEATURES / f"{ticker}_features.csv")
            all_frames.append(df)
        else:
            failed.append(ticker)

    if not all_frames:
        raise RuntimeError("No se generaron features. Ejecuta data_curator.py primero.")

    df_all = pd.concat(all_frames)
    df_all.to_csv(DATA_FEATURES / "_all_features.csv")

    log.info(f"\n{'='*50}")
    log.info(f"  ✅ Exitosos : {len(all_frames)}")
    log.info(f"  ❌ Fallidos : {len(failed)} {failed if failed else ''}")
    log.info(f"  📦 Consolidado: {DATA_FEATURES / '_all_features.csv'}")
    log.info(f"{'='*50}")

    return df_all


if __name__ == "__main__":
    engineer_features()
