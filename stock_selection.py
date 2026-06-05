"""
stock_selection.py — Selección de acciones Top-N
=================================================
1. Lee predictions.csv (scores por ticker por fecha)
2. Identifica fechas de rebalanceo (viernes)
3. En cada fecha de rebalanceo, ordena tickers por score descendente
4. Selecciona únicamente los Top-N con señal BUY
5. Guarda selection.csv con los tickers seleccionados por fecha
"""

import logging
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_SIGNALS, DATA_PORTFOLIOS,
    TICKERS, TOP_N_STOCKS,
    START_DATE, END_DATE,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PREDICTIONS_PATH = DATA_SIGNALS / "predictions.csv"
SELECTION_OUT    = DATA_PORTFOLIOS / "selection.csv"


def get_rebalance_dates(start: str, end: str, trading_dates: pd.DatetimeIndex = None) -> pd.DatetimeIndex:
    """
    Genera fechas de rebalanceo (viernes) usando solo días hábiles reales del mercado.
    Si el viernes es festivo, retrocede al día hábil anterior.
    """
    all_fridays = pd.date_range(start=start, end=end, freq="W-FRI")

    if trading_dates is None:
        log.info(f"  Fechas de rebalanceo: {len(all_fridays)} viernes")
        return all_fridays

    trading_set = set(trading_dates.normalize())
    valid_dates = []
    for friday in all_fridays:
        candidate = friday
        # Retroceder hasta encontrar un día hábil real
        for _ in range(7):
            if pd.Timestamp(candidate).normalize() in trading_set:
                valid_dates.append(candidate)
                break
            candidate -= pd.Timedelta(days=1)

    result = pd.DatetimeIndex(sorted(set(valid_dates)))
    log.info(f"  Fechas de rebalanceo: {len(result)} días hábiles (de {len(all_fridays)} viernes)")
    return result


def select_top_n(signals: pd.DataFrame, rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Para cada fecha de rebalanceo:
    - Busca señales disponibles en esa fecha (o el día hábil más cercano anterior)
    - Filtra solo tickers del universo con señal BUY
    - Ordena por score descendente
    - Toma los Top-N
    """
    # Asegurar que el índice es DatetimeIndex
    signals.index = pd.to_datetime(signals.index)

    # Pivotear: índice=Date, columnas=Ticker, valores=score
    score_pivot  = signals.pivot_table(index=signals.index, columns="Ticker", values="score")
    signal_pivot = signals.pivot_table(index=signals.index, columns="Ticker", values="signal", aggfunc="first")

    # Solo tickers del universo
    universe_cols = [t for t in TICKERS if t in score_pivot.columns]
    score_pivot   = score_pivot[universe_cols]
    signal_pivot  = signal_pivot[universe_cols] if all(c in signal_pivot.columns for c in universe_cols) else signal_pivot

    selection_records = []

    for rebal_date in rebalance_dates:
        # Buscar fecha disponible más cercana (hacia atrás)
        available = score_pivot.index[score_pivot.index <= rebal_date]
        if len(available) == 0:
            continue
        closest_date = available[-1]

        scores_day = score_pivot.loc[closest_date].dropna()

        # Filtrar solo BUY si tenemos señales
        if closest_date in signal_pivot.index:
            signals_day = signal_pivot.loc[closest_date]
            buy_mask    = signals_day == "BUY"
            scores_day  = scores_day[buy_mask.reindex(scores_day.index, fill_value=False)]

        if scores_day.empty:
            log.warning(f"  ⚠ {rebal_date.date()}: sin señales BUY disponibles, omitiendo")
            continue

        # Top-N por score
        top_n = scores_day.nlargest(TOP_N_STOCKS)

        for rank, (ticker, score) in enumerate(top_n.items(), 1):
            selection_records.append({
                "rebalance_date": rebal_date.date(),
                "data_date":      closest_date.date(),
                "ticker":         ticker,
                "score":          round(score, 6),
                "rank":           rank,
            })

    df_selection = pd.DataFrame(selection_records)
    log.info(f"\n{'='*50}")
    log.info(f"  ✅ Fechas de rebalanceo procesadas: {df_selection['rebalance_date'].nunique()}")
    log.info(f"  📋 Total selecciones: {len(df_selection)}")
    log.info(f"  🏆 Top tickers más frecuentes:\n{df_selection['ticker'].value_counts().head(10).to_string()}")
    log.info(f"{'='*50}")

    return df_selection


def run_selection() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"No encontrado: {PREDICTIONS_PATH}. Ejecuta model_xgboost.py primero.")

    signals = pd.read_csv(PREDICTIONS_PATH, index_col="Date", parse_dates=True)
    log.info(f"Señales cargadas: {signals.shape}")

    # Usar fechas reales de trading para evitar festivos
    trading_dates = signals.index
    rebalance_dates = get_rebalance_dates(START_DATE, END_DATE, trading_dates)
    selection = select_top_n(signals, rebalance_dates)

    selection.to_csv(SELECTION_OUT, index=False)
    log.info(f"  💾 Selección guardada: {SELECTION_OUT}")

    return selection


if __name__ == "__main__":
    run_selection()
