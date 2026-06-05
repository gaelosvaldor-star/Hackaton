"""
config.py — Configuración central del pipeline
Hackaton | Pipeline 100% Python
"""

from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
DATA_RAW        = Path(r"C:\Users\gaelo\OneDrive\Escritorio\Hakaton\Output")  # CSVs de KaxaNuk
DATA_FEATURES   = BASE_DIR / "data" / "features"
DATA_SIGNALS    = BASE_DIR / "data" / "signals"
DATA_PORTFOLIOS = BASE_DIR / "data" / "portfolios"
DATA_BACKTEST   = BASE_DIR / "data" / "backtest"
MODELS_DIR      = BASE_DIR / "models"

for d in [DATA_FEATURES, DATA_SIGNALS, DATA_PORTFOLIOS, DATA_BACKTEST, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Periodo histórico ─────────────────────────────────────────────
START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"

# ── Universo de activos (sin SPY, SPY es solo benchmark) ──────────
TICKERS = [
    "NVDA", "GOOG", "META", "IBKR", "MSFT",
    "AAPL", "SCHW", "AXP", "ORCL", "INTC",
    "F", "GM", "APO", "JNJ", "BA",
    "SYF", "BN", "KKR", "KDP", "COST",
    "MELI", "CSCO", "BKNG", "BKR", "RJF",
    "MRX", "V", "MU", "AVGO", "UAL",
    "KO", "GEF", "PFE", "HON", "AMD",
    "ANET", "PG", "T", "NFLX", "ABNB",
    "AMGN", "FDX", "CRM", "IBM", "CVS",
    "DEL", "GE", "PEP", "COIN", "WMT",
]

BENCHMARK = "SPY"

# ── Capital inicial ───────────────────────────────────────────────
INITIAL_CAPITAL = 100_000

# ── Rebalanceo ────────────────────────────────────────────────────
REBALANCE_FREQUENCY = "weekly"    # semanal (viernes)
TOP_N_STOCKS        = 15          # activos seleccionados por rebalanceo

# ── Costos de transacción ─────────────────────────────────────────
COMMISSION_RATE = 0.001           # 0.1% por operación
SPREAD          = 0.001           # 0.1% spread bid/ask

# ── Columnas clave de KaxaNuk ─────────────────────────────────────
COL_DATE   = "m_date"
COL_CLOSE  = "m_close_dividend_and_split_adjusted"
COL_OPEN   = "m_open_dividend_and_split_adjusted"
COL_HIGH   = "m_high_dividend_and_split_adjusted"
COL_LOW    = "m_low_dividend_and_split_adjusted"
COL_VOLUME = "m_volume_split_adjusted"

# Columnas de KaxaNuk ya calculadas que usaremos como features
KAXANUK_FEATURES = [
    "c_rsi_14d_dividend_and_split_adjusted",
    "c_annualized_volatility_21d_log_returns_dividend_and_split_adjusted",
    "c_annualized_volatility_63d_log_returns_dividend_and_split_adjusted",
    "c_simple_moving_average_21d_close_dividend_and_split_adjusted",
    "c_simple_moving_average_63d_close_dividend_and_split_adjusted",
    "c_exponential_moving_average_21d_close_dividend_and_split_adjusted",
    "c_macd_26d_12d_dividend_and_split_adjusted",
    "c_macd_signal_9d_dividend_and_split_adjusted",
    "c_log_returns_dividend_and_split_adjusted",
]

# ── Parámetros de Features adicionales ───────────────────────────
RETURNS_PERIODS  = [1, 5, 21, 63]
VOL_WINDOWS      = [21, 63]
RSI_WINDOW       = 14
SMA_WINDOW       = 50
MOMENTUM_WINDOWS = [10, 21]
VOL_AVG_WINDOW   = 20

# ── XGBoost ───────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators":    200,
    "max_depth":       4,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "random_state":    42,
    "n_jobs":         -1,
}
TRAIN_TEST_SPLIT = 0.8            # 80% train, 20% test
FORWARD_RETURN_DAYS = 5           # predecir retorno a 5 días (1 semana)

# ── Markowitz ─────────────────────────────────────────────────────
MIN_WEIGHT = 0.01                 # peso mínimo por activo (1%)
MAX_WEIGHT = 0.20                 # peso máximo por activo (20%)

