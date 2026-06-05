"""
markowitz_optimizer.py — Optimización de Portafolio (Markowitz)
================================================================
Para cada fecha de rebalanceo:
1. Toma los Top-N tickers seleccionados
2. Calcula retornos históricos (ventana de 63 días)
3. Calcula matriz de covarianzas
4. Optimiza pesos maximizando Sharpe Ratio (scipy minimize)
5. Aplica restricciones: peso mín/máx por activo, suma = 1
6. Guarda portfolio_weights.csv en formato wide (tickers × fechas)
   compatible con kaxanuk.backtest_engine
"""

import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_FEATURES, DATA_PORTFOLIOS,
    TICKERS,
    MIN_WEIGHT, MAX_WEIGHT,
    START_DATE, END_DATE,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SELECTION_PATH   = DATA_PORTFOLIOS / "selection.csv"
WEIGHTS_OUT      = DATA_PORTFOLIOS / "portfolio_weights.csv"
LOOKBACK_DAYS    = 63    # ventana histórica para covarianza (1 trimestre)
RISK_FREE_RATE   = 0.05  # tasa libre de riesgo anual (~Fed Funds)


def load_prices() -> pd.DataFrame:
    """
    Carga precios de cierre de todos los tickers desde sus CSVs de features.
    Devuelve DataFrame wide: índice=Date, columnas=Ticker.
    """
    frames = {}
    for ticker in TICKERS:
        path = DATA_FEATURES / f"{ticker}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        if "Close" in df.columns:
            frames[ticker] = df["Close"]

    prices = pd.DataFrame(frames)
    prices = prices.sort_index()
    log.info(f"Precios cargados: {prices.shape[0]} fechas | {prices.shape[1]} tickers")
    return prices


def sharpe_objective(weights: np.ndarray, mean_returns: np.ndarray,
                     cov_matrix: np.ndarray, rf_daily: float) -> float:
    """Negativo del Sharpe Ratio (minimizamos el negativo = maximizamos Sharpe)."""
    port_return = np.dot(weights, mean_returns) * 252
    port_vol    = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(252)
    sharpe      = (port_return - RISK_FREE_RATE) / (port_vol + 1e-9)
    return -sharpe


def optimize_weights(tickers: list[str], prices: pd.DataFrame,
                     ref_date: pd.Timestamp) -> dict[str, float] | None:
    """
    Optimiza pesos para un conjunto de tickers en una fecha dada.
    Usa los últimos LOOKBACK_DAYS días de retornos.
    """
    # Ventana histórica
    hist = prices.loc[:ref_date, tickers].dropna(how="all")
    hist = hist.tail(LOOKBACK_DAYS).dropna(axis=1)

    valid_tickers = hist.columns.tolist()
    if len(valid_tickers) < 2:
        log.warning(f"  ⚠ {ref_date.date()}: menos de 2 tickers con datos, skip")
        return None

    returns      = hist.pct_change().dropna()
    mean_returns = returns.mean().values
    cov_matrix   = returns.cov().values
    n            = len(valid_tickers)
    rf_daily     = RISK_FREE_RATE / 252

    # Restricciones
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds      = [(MIN_WEIGHT, MAX_WEIGHT)] * n
    w0          = np.array([1.0 / n] * n)   # pesos iniciales iguales

    result = minimize(
        sharpe_objective,
        w0,
        args=(mean_returns, cov_matrix, rf_daily),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if not result.success:
        log.warning(f"  ⚠ {ref_date.date()}: optimización no convergió → pesos iguales")
        raw_weights = [1.0 / n] * n
    else:
        raw_weights = list(result.x)

    # Normalizar para garantizar suma = 1 exacto
    total = sum(raw_weights)
    raw_weights = [w / total for w in raw_weights]

    # Ajustar último peso para eliminar error de punto flotante
    diff = 1.0 - sum(raw_weights)
    raw_weights[-1] += diff

    weights = {t: round(w, 8) for t, w in zip(valid_tickers, raw_weights)}

    return weights


def build_weights_matrix(selection: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la matriz de pesos en formato wide:
    - Índice (filas)   : Ticker
    - Columnas         : fechas de rebalanceo (dd/mm/yyyy)
    - Valores          : pesos decimales (0 si no seleccionado)
    """
    all_tickers     = sorted(TICKERS)
    rebalance_dates = sorted(selection["rebalance_date"].unique())
    weights_dict    = {}   # fecha → {ticker: peso}

    for rebal_date in rebalance_dates:
        ref_ts  = pd.Timestamp(rebal_date)
        day_sel = selection[selection["rebalance_date"] == rebal_date]
        tickers = day_sel["ticker"].tolist()

        opt_weights = optimize_weights(tickers, prices, ref_ts)

        if opt_weights is None:
            # Pesos iguales como fallback
            n = len(tickers)
            opt_weights = {t: round(1.0 / n, 6) for t in tickers} if n > 0 else {}

        # Rellenar todos los tickers con 0
        full_weights = {t: 0.0 for t in all_tickers}
        full_weights.update(opt_weights)

        # Formato de fecha compatible con kaxanuk (dd/mm/yyyy)
        col_name = ref_ts.strftime("%d/%m/%Y")
        weights_dict[col_name] = full_weights

        log.info(f"  ✓ {rebal_date} → {len(opt_weights)} activos | "
                 f"Sharpe proxy OK")

    df_weights = pd.DataFrame(weights_dict)
    df_weights.index.name = "Ticker"
    df_weights = df_weights.loc[all_tickers]   # orden consistente

    return df_weights


def run_optimizer() -> pd.DataFrame:
    if not SELECTION_PATH.exists():
        raise FileNotFoundError(f"No encontrado: {SELECTION_PATH}. Ejecuta stock_selection.py primero.")

    selection = pd.read_csv(SELECTION_PATH)
    log.info(f"Selección cargada: {len(selection)} registros | "
             f"{selection['rebalance_date'].nunique()} fechas")

    prices = load_prices()

    log.info("Optimizando pesos (Markowitz - Max Sharpe)...")
    weights_matrix = build_weights_matrix(selection, prices)

    weights_matrix.to_csv(WEIGHTS_OUT)
    log.info(f"\n{'='*50}")
    log.info(f"  ✅ Matriz de pesos: {weights_matrix.shape}")
    log.info(f"  💾 Guardada en: {WEIGHTS_OUT}")
    log.info(f"{'='*50}")

    return weights_matrix


if __name__ == "__main__":
    run_optimizer()
