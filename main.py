"""
main.py — Entry Point del Pipeline
====================================
Ejecuta el pipeline completo en orden:
  1. data_curator    → Lee y limpia CSVs de KaxaNuk
  2. features        → Feature engineering
  3. model_xgboost   → Entrena modelo y genera señales
  4. stock_selection → Selecciona Top-N por fecha
  5. markowitz       → Optimiza pesos del portafolio
  6. backtest        → Conecta con kaxanuk.backtest_engine
  7. metrics         → Calcula y reporta KPIs

Uso:
  python main.py                    # pipeline completo
  python main.py --skip-backtest    # sin backtest engine (solo métricas propias)
  python main.py --from features    # desde un paso específico
"""

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

STEPS = [
    "curator",
    "features",
    "model",
    "selection",
    "markowitz",
    "backtest",
    "metrics",
]


def run_step(name: str, fn, *args, **kwargs):
    log.info(f"\n{'█'*55}")
    log.info(f"  PASO: {name.upper()}")
    log.info(f"{'█'*55}")
    t0     = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - t0
    log.info(f"  ✅ {name} completado en {elapsed:.1f}s")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline Hackaton - Portafolio Cuantitativo")
    parser.add_argument(
        "--from", dest="from_step",
        choices=STEPS,
        default=None,
        help="Iniciar desde un paso específico (omite los anteriores)",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Omitir el paso de backtest engine (útil si no está configurado aún)",
    )
    parser.add_argument(
        "--only-metrics",
        action="store_true",
        help="Solo calcular métricas (asume que ya tienes equity_curve y pesos)",
    )
    return parser.parse_args()


def main():
    args       = parse_args()
    from_step  = args.from_step
    skip_steps = set()

    if args.only_metrics:
        from_step = "metrics"

    if from_step:
        idx = STEPS.index(from_step)
        skip_steps = set(STEPS[:idx])
        log.info(f"▶ Iniciando desde: {from_step.upper()} (omitiendo: {skip_steps})")

    t_total = time.time()

    # ── 1. Data Curator ───────────────────────────────────────────
    if "curator" not in skip_steps:
        from data_curator import curate_all
        run_step("Data Curator", curate_all)

    # ── 2. Features ───────────────────────────────────────────────
    if "features" not in skip_steps:
        from features import engineer_features
        run_step("Feature Engineering", engineer_features)

    # ── 3. Modelo XGBoost ─────────────────────────────────────────
    if "model" not in skip_steps:
        from model_xgboost import run_model
        signals = run_step("XGBoost Model", run_model)

    # ── 4. Selección de acciones ──────────────────────────────────
    if "selection" not in skip_steps:
        from stock_selection import run_selection
        selection = run_step("Stock Selection", run_selection)

    # ── 5. Markowitz ──────────────────────────────────────────────
    if "markowitz" not in skip_steps:
        from markowitz_optimizer import run_optimizer
        weights = run_step("Markowitz Optimizer", run_optimizer)

    # ── 6. Backtest Engine ────────────────────────────────────────
    if "backtest" not in skip_steps and not args.skip_backtest:
        try:
            from backtest_adapter import run_adapter
            run_step("Backtest Engine", run_adapter)
        except Exception as e:
            log.warning(f"  ⚠ Backtest engine falló: {e}")
            log.warning("  Continuando con métricas propias...")

    # ── 7. Métricas ───────────────────────────────────────────────
    if "metrics" not in skip_steps:
        from metrics import run_metrics
        metrics, equity = run_step("Metrics", run_metrics)

    # ── Resumen final ─────────────────────────────────────────────
    total_time = time.time() - t_total
    log.info(f"\n{'='*55}")
    log.info(f"  🏁 PIPELINE COMPLETADO en {total_time:.1f}s")
    log.info(f"{'='*55}")


if __name__ == "__main__":
    main()
