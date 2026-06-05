"""
metrics.py — Métricas y KPIs del portafolio
============================================
Calcula e imprime los KPIs principales comparando
la estrategia vs el benchmark SPY:
  - Retorno acumulado
  - Volatilidad anualizada
  - Sharpe Ratio
  - Drawdown máximo
  - Alpha y Beta
  - Turnover promedio
  - Calmar Ratio
Genera backtest_report.csv y equity_curve.csv
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_FEATURES, DATA_PORTFOLIOS, DATA_BACKTEST,
    BENCHMARK,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RISK_FREE_RATE   = 0.05          # tasa anual libre de riesgo
REPORT_OUT       = DATA_BACKTEST / "backtest_report.csv"
EQUITY_CURVE_OUT = DATA_BACKTEST / "equity_curve.csv"
WEIGHTS_PATH     = DATA_PORTFOLIOS / "portfolio_weights.csv"


# ── Funciones de métricas ─────────────────────────────────────────

def cumulative_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def annualized_return(returns: pd.Series) -> float:
    n_years = len(returns) / 252
    return float((1 + cumulative_return(returns)) ** (1 / n_years) - 1) if n_years > 0 else 0.0


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(252))


def sharpe_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    excess = annualized_return(returns) - rf
    vol    = annualized_volatility(returns)
    return float(excess / vol) if vol > 0 else 0.0


def max_drawdown(returns: pd.Series) -> float:
    equity   = (1 + returns).cumprod()
    peak     = equity.cummax()
    drawdown = (equity - peak) / peak
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series) -> float:
    ann_ret = annualized_return(returns)
    mdd     = abs(max_drawdown(returns))
    return float(ann_ret / mdd) if mdd > 0 else 0.0


def alpha_beta(strategy_returns: pd.Series, benchmark_returns: pd.Series,
               rf: float = RISK_FREE_RATE) -> tuple[float, float]:
    rf_daily   = rf / 252
    excess_s   = strategy_returns  - rf_daily
    excess_b   = benchmark_returns - rf_daily
    cov_matrix = np.cov(excess_s.dropna(), excess_b.dropna())
    beta       = float(cov_matrix[0, 1] / cov_matrix[1, 1]) if cov_matrix[1, 1] != 0 else 0.0
    alpha_ann  = float((annualized_return(strategy_returns) - rf)
                       - beta * (annualized_return(benchmark_returns) - rf))
    return alpha_ann, beta


def build_equity_curve(weights_matrix: pd.DataFrame,
                       prices: pd.DataFrame,
                       initial_capital: float = 100_000) -> pd.DataFrame:
    """
    Construye la equity curve a partir de la matriz de pesos.
    Asume que los pesos se mantienen hasta el siguiente rebalanceo.
    """
    # Parsear fechas de columnas (formato dd/mm/yyyy)
    rebal_dates = pd.to_datetime(weights_matrix.columns, format="%d/%m/%Y")
    weights_matrix.columns = rebal_dates

    # Rango de fechas completo
    all_dates = prices.index
    portfolio_values = []

    current_weights = None
    capital = initial_capital

    for i, date in enumerate(all_dates):
        # Actualizar pesos si es fecha de rebalanceo
        applicable = rebal_dates[rebal_dates <= date]
        if len(applicable) > 0:
            latest_rebal = applicable[-1]
            w_series = weights_matrix[latest_rebal]
            current_weights = w_series[w_series > 0]

        if current_weights is None:
            portfolio_values.append(capital)
            continue

        # Calcular retorno del día
        tickers = current_weights.index.tolist()
        available = [t for t in tickers if t in prices.columns and not pd.isna(prices.loc[date, t])]
        if not available:
            portfolio_values.append(capital if not portfolio_values else portfolio_values[-1])
            continue

        weights_valid = current_weights[available]
        weights_valid = weights_valid / weights_valid.sum()

        if i == 0:
            daily_return = 0.0
        else:
            prev_date = all_dates[i - 1]
            day_returns = (prices.loc[date, available] / prices.loc[prev_date, available] - 1).fillna(0)
            daily_return = float((weights_valid * day_returns).sum())

        capital = capital * (1 + daily_return)
        portfolio_values.append(capital)

    equity_curve = pd.Series(portfolio_values, index=all_dates, name="Portfolio")
    return equity_curve.to_frame()


def compute_metrics(strategy_returns: pd.Series,
                    benchmark_returns: pd.Series) -> dict:
    alpha, beta = alpha_beta(strategy_returns, benchmark_returns)

    metrics = {
        "Retorno Acumulado (Estrategia)":   f"{cumulative_return(strategy_returns):.2%}",
        "Retorno Acumulado (SPY)":          f"{cumulative_return(benchmark_returns):.2%}",
        "Retorno Anualizado (Estrategia)":  f"{annualized_return(strategy_returns):.2%}",
        "Retorno Anualizado (SPY)":         f"{annualized_return(benchmark_returns):.2%}",
        "Volatilidad Anualizada":           f"{annualized_volatility(strategy_returns):.2%}",
        "Sharpe Ratio":                     f"{sharpe_ratio(strategy_returns):.4f}",
        "Sharpe Ratio (SPY)":               f"{sharpe_ratio(benchmark_returns):.4f}",
        "Max Drawdown":                     f"{max_drawdown(strategy_returns):.2%}",
        "Max Drawdown (SPY)":               f"{max_drawdown(benchmark_returns):.2%}",
        "Calmar Ratio":                     f"{calmar_ratio(strategy_returns):.4f}",
        "Alpha (anualizado)":               f"{alpha:.2%}",
        "Beta":                             f"{beta:.4f}",
    }
    return metrics


def run_metrics(equity_curve: pd.DataFrame = None):
    """
    Calcula métricas. Si no se pasa equity_curve, intenta reconstruirla
    desde los pesos y precios guardados.
    """
    # Cargar precios
    spy_path = DATA_FEATURES / "SPY.csv"
    if not spy_path.exists():
        log.warning("SPY.csv no encontrado, usando precios del consolidado")

    frames = {}
    from config import TICKERS
    for ticker in TICKERS + [BENCHMARK]:
        p = DATA_FEATURES / f"{ticker}.csv"
        if p.exists():
            df = pd.read_csv(p, index_col="Date", parse_dates=True)
            if "Close" in df.columns:
                frames[ticker] = df["Close"]

    prices = pd.DataFrame(frames).sort_index()

    # Construir equity curve si no viene del backtest engine
    if equity_curve is None:
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"No encontrado: {WEIGHTS_PATH}")
        weights_matrix = pd.read_csv(WEIGHTS_PATH, index_col="Ticker")
        log.info("Construyendo equity curve desde pesos...")
        equity_df = build_equity_curve(weights_matrix, prices)
    else:
        equity_df = equity_curve

    # Retornos diarios
    strategy_returns  = equity_df["Portfolio"].pct_change().dropna()
    benchmark_returns = prices[BENCHMARK].pct_change().dropna()

    # Alinear fechas
    common_idx        = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_returns  = strategy_returns.loc[common_idx]
    benchmark_returns = benchmark_returns.loc[common_idx]

    # Calcular métricas
    metrics = compute_metrics(strategy_returns, benchmark_returns)

    # Imprimir reporte
    log.info(f"\n{'='*55}")
    log.info("  📊 REPORTE DE BACKTEST")
    log.info(f"{'='*55}")
    for k, v in metrics.items():
        log.info(f"  {k:<40} {v}")
    log.info(f"{'='*55}")

    # Guardar reporte CSV
    pd.DataFrame(list(metrics.items()), columns=["Métrica", "Valor"]).to_csv(REPORT_OUT, index=False)
    log.info(f"  💾 Reporte guardado: {REPORT_OUT}")

    # Guardar equity curve
    spy_equity = (1 + benchmark_returns).cumprod() * 100_000
    equity_out = equity_df.copy()
    equity_out["SPY"] = spy_equity
    equity_out.to_csv(EQUITY_CURVE_OUT)
    log.info(f"  💾 Equity curve guardada: {EQUITY_CURVE_OUT}")

    return metrics, equity_out


if __name__ == "__main__":
    run_metrics()
