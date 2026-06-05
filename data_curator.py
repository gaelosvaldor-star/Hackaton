"""
data_curator.py — Lee los CSVs de KaxaNuk y los prepara para el pipeline
=========================================================================
- Lee cada CSV de KaxaNuk desde DATA_RAW
- Filtra el periodo START_DATE → END_DATE
- Selecciona columnas relevantes (precios + features KaxaNuk)
- Guarda un CSV limpio por ticker en DATA_FEATURES
- Genera _all_data.csv consolidado
"""

import logging
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_RAW, DATA_FEATURES,
    TICKERS, BENCHMARK,
    START_DATE, END_DATE,
    COL_DATE, COL_CLOSE, COL_OPEN, COL_HIGH, COL_LOW, COL_VOLUME,
    KAXANUK_FEATURES,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Columnas base de precio que siempre necesitamos
BASE_COLS = [COL_DATE, COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME]


def load_ticker(ticker: str) -> pd.DataFrame | None:
    """
    Carga un CSV de KaxaNuk, filtra fechas y selecciona columnas relevantes.
    """
    path = DATA_RAW / f"{ticker}.csv"
    if not path.exists():
        log.warning(f"  ✗ {ticker}: archivo no encontrado en {path}")
        return None

    df = pd.read_csv(path, low_memory=False)

    # Verificar columna de fecha
    if COL_DATE not in df.columns:
        log.warning(f"  ✗ {ticker}: columna '{COL_DATE}' no encontrada")
        return None

    # Parsear y filtrar fechas
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df = df.dropna(subset=[COL_DATE])
    df = df[(df[COL_DATE] >= START_DATE) & (df[COL_DATE] <= END_DATE)]
    df = df.sort_values(COL_DATE).reset_index(drop=True)

    if df.empty:
        log.warning(f"  ✗ {ticker}: sin datos en el periodo {START_DATE} → {END_DATE}")
        return None

    # Seleccionar columnas disponibles
    available_features = [c for c in KAXANUK_FEATURES if c in df.columns]
    cols_to_keep = BASE_COLS + available_features
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]

    df = df[cols_to_keep].copy()

    # Renombrar columnas para simplificar uso posterior
    df = df.rename(columns={
        COL_DATE:   "Date",
        COL_CLOSE:  "Close",
        COL_OPEN:   "Open",
        COL_HIGH:   "High",
        COL_LOW:    "Low",
        COL_VOLUME: "Volume",
    })

    df["Ticker"] = ticker
    df = df.set_index("Date")

    # Eliminar filas donde Close es nulo (sin precio)
    df = df.dropna(subset=["Close"])

    log.info(f"  ✓ {ticker}: {len(df):,} filas | {len(df.columns)} columnas")
    return df


def curate_all() -> pd.DataFrame:
    """
    Procesa todos los tickers + benchmark.
    Guarda CSVs individuales y un consolidado.
    """
    all_tickers = TICKERS + ([BENCHMARK] if BENCHMARK not in TICKERS else [])
    all_frames  = []
    failed      = []

    log.info(f"Procesando {len(all_tickers)} tickers...")

    for ticker in all_tickers:
        df = load_ticker(ticker)
        if df is not None:
            df.to_csv(DATA_FEATURES / f"{ticker}.csv")
            all_frames.append(df)
        else:
            failed.append(ticker)

    if not all_frames:
        raise RuntimeError("No se cargó ningún ticker. Verifica DATA_RAW.")

    df_all = pd.concat(all_frames)
    consolidated_path = DATA_FEATURES / "_all_data.csv"
    df_all.to_csv(consolidated_path)

    log.info(f"\n{'='*50}")
    log.info(f"  ✅ Exitosos : {len(all_frames)}")
    log.info(f"  ❌ Fallidos : {len(failed)} {failed if failed else ''}")
    log.info(f"  📦 Consolidado: {consolidated_path}")
    log.info(f"{'='*50}")

    return df_all


if __name__ == "__main__":
    curate_all()
