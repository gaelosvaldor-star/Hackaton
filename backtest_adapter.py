"""
backtest_adapter.py — Adaptador para kaxanuk.backtest_engine
=============================================================
1. Copia los datos de mercado al directorio Input/Data/
2. Copia portfolio_weights.csv al directorio Input/Portfolios/
3. Ejecuta el backtest con PyArrowBacktester
4. Guarda resultados en Output/
"""

import logging
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BASE_DIR, DATA_RAW, DATA_PORTFOLIOS,
    TICKERS, BENCHMARK,
    START_DATE, END_DATE,
    INITIAL_CAPITAL, COMMISSION_RATE, SPREAD,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Directorios del backtest engine ──────────────────────────────
BACKTEST_ROOT        = BASE_DIR / "backtest_engine"
CONFIG_DIR           = BACKTEST_ROOT / "Config"
INPUT_DATA_DIR       = BACKTEST_ROOT / "Input" / "Data"
INPUT_PORTFOLIO_DIR  = BACKTEST_ROOT / "Input" / "Portfolios"
OUTPUT_DIR           = BACKTEST_ROOT / "Output"
ENV_FILE             = CONFIG_DIR / ".env"
EXCEL_CONFIG         = CONFIG_DIR / "backtest_engine_parameters.xlsx"
WEIGHTS_SOURCE       = DATA_PORTFOLIOS / "portfolio_weights.csv"

for d in [CONFIG_DIR, INPUT_DATA_DIR, INPUT_PORTFOLIO_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def copy_market_data():
    """Copia los CSVs de KaxaNuk al directorio Input/Data/ del backtest engine."""
    all_tickers = TICKERS + ([BENCHMARK] if BENCHMARK not in TICKERS else [])
    copied  = 0
    missing = []

    for ticker in all_tickers:
        src = DATA_RAW / f"{ticker}.csv"
        dst = INPUT_DATA_DIR / f"{ticker}.csv"
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing.append(ticker)

    log.info(f"  ✓ Datos de mercado copiados: {copied} archivos")
    if missing:
        log.warning(f"  ⚠ Faltantes: {missing}")


def copy_portfolio_weights():
    """Copia portfolio_weights.csv al directorio Input/Portfolios/."""
    if not WEIGHTS_SOURCE.exists():
        raise FileNotFoundError(
            f"No encontrado: {WEIGHTS_SOURCE}. Ejecuta markowitz_optimizer.py primero."
        )
    dst = INPUT_PORTFOLIO_DIR / "portfolio_weights.csv"
    shutil.copy2(WEIGHTS_SOURCE, dst)
    log.info(f"  ✓ Pesos copiados a: {dst}")
    return dst


def run_backtest():
    """
    Ejecuta el backtest usando kaxanuk.backtest_engine.
    Requiere:
      - Config/backtest_engine_parameters.xlsx configurado
      - Config/.env con KNBE_API_KEY_KAXANUK
    """
    try:
        from kaxanuk.backtest_engine.backtest.pyarrow_backtester import PyArrowBacktester
        from kaxanuk.backtest_engine.config_handlers.excel_configurator import ExcelConfigurator
        from kaxanuk.backtest_engine.data_processors.data_pipeline_executor import execute_data_pipeline
        from kaxanuk.backtest_engine.input_handlers.csv_input import CsvInput
        from kaxanuk.backtest_engine.input_handlers.csv_portfolio_input_handler import CsvPortfolioInputHandler
        from kaxanuk.backtest_engine.services.config_logger import configure_logger
        from kaxanuk.backtest_engine.services.env_loader import load_config_env
        from kaxanuk.backtest_engine.exceptions import BacktestError, DataPipelineError, ConfigurationError
    except ImportError as e:
        log.error(f"kaxanuk.backtest_engine no instalado: {e}")
        raise

    # Cargar .env con API key
    load_config_env(ENV_FILE)

    # Verificar que existe el Excel de configuración
    if not EXCEL_CONFIG.exists():
        raise FileNotFoundError(
            f"No encontrado: {EXCEL_CONFIG}\n"
            f"Ejecuta: kaxanuk.backtest_engine init excel\n"
            f"y configura el archivo Excel con los parámetros del backtest."
        )

    # 1. Configuración
    configurator = ExcelConfigurator(file_path=str(EXCEL_CONFIG))
    try:
        configuration = configurator.get_configuration()
    except ConfigurationError as ex:
        log.error(f"Error de configuración: {ex}")
        raise

    log.info(f"  ✓ Configuración cargada: {configuration.portfolio_name}")

    # 2. Input handlers
    market_data_input = CsvInput(input_dir=str(INPUT_DATA_DIR))
    portfolio_input   = CsvPortfolioInputHandler(base_dir=str(INPUT_PORTFOLIO_DIR))

    # 3. Pipeline de datos
    try:
        pipeline_result = execute_data_pipeline(
            configuration=configuration,
            input_handlers=[market_data_input],
            portfolio_handlers=[portfolio_input],
        )
    except DataPipelineError as ex:
        log.error(f"Error en pipeline de datos: {ex}")
        raise

    log.info("  ✓ Pipeline de datos ejecutado")

    # 4. Backtester
    backtester = PyArrowBacktester.create_from_pipeline_result(
        configuration=configuration,
        pipeline_result=pipeline_result,
    )

    # 5. Ejecutar backtest vs benchmark (SPY)
    try:
        results = backtester.run_with_benchmark()
    except BacktestError as ex:
        log.error(f"Error en backtest: {ex}")
        raise

    log.info(f"\n{'='*50}")
    log.info("  ✅ Backtest completado")
    log.info(f"  📊 Resultados en: {OUTPUT_DIR}")
    log.info(f"{'='*50}")

    return results


def run_adapter():
    log.info("=== Preparando backtest engine ===")

    log.info("1. Copiando datos de mercado...")
    copy_market_data()

    log.info("2. Copiando pesos del portafolio...")
    copy_portfolio_weights()

    log.info("3. Ejecutando backtest...")
    results = run_backtest()

    return results


if __name__ == "__main__":
    run_adapter()
