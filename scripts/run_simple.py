"""
scripts/run_simple.py
One-click runner for the simplified IEEE 34-bus OPF case (no Jupyter required).

Usage:
    python scripts/run_simple.py
    python scripts/run_simple.py --data "data/900131_*.csv" --epochs 100
"""

import argparse
import sys
import os
import yaml
import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset      import load_and_preprocess_all
from src.model        import QRLSTM
from src.train        import train_model, evaluate
from src.opf_simple   import solve_opf, P_LOAD_TOTAL
from src.monte_carlo  import run_mc, summarize
from src              import utils


def load_config(path="configs/simple_ieee34.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main(args):
    cfg = load_config(args.config)
    rc  = cfg["renewable"]
    mc  = cfg["monte_carlo"]
    tr  = cfg["training"]
    opf = cfg["opf"]

    # Renewable energy parameters
    pv_mu      = rc["pv_mu"]
    wind_mu    = rc["wind_mu"]
    sigma_pv   = rc["sigma_ratio"] * pv_mu
    sigma_wind = rc["sigma_ratio"] * wind_mu
    k          = rc["robust_sigma"]
    pv_low     = max(0.0, pv_mu   - k * sigma_pv)
    wind_low   = max(0.0, wind_mu - k * sigma_wind)

    print("=== Simple IEEE 34-bus OPF ===")
    print(f"P_LOAD_TOTAL = {P_LOAD_TOTAL:.2f} MW")
    print(f"PV   mu={pv_mu:.3f}  sigma={sigma_pv:.3f}")
    print(f"Wind mu={wind_mu:.3f}  sigma={sigma_wind:.3f}\n")

    # Step 1: train QRLSTM (only if data is provided)
    if args.data:
        print("=== Step 1: Train QRLSTM ===")
        X_train, X_test, y_train, y_test, scaler, target_indices = \
            load_and_preprocess_all(args.data, window_size=tr["window_size"])
        model = QRLSTM(X_train.shape[2], hidden_size=tr["hidden_size"])
        model, loss_history = train_model(
            model, X_train, y_train,
            epochs=args.epochs or tr["epochs"],
            batch_size=tr["batch_size"],
            lr=tr["lr"],
        )
        utils.plot_quantile_predictions(model, X_test, y_test, scaler, target_indices)
    else:
        print("(Skipping QRLSTM training — no --data provided)\n")

    # Step 2: OPF planning
    print("=== Step 2: OPF Planning ===")
    results = {}
    for mode, label in [("det", "Deterministic"), ("ccopf", "CCOPF"), ("robust", "Robust")]:
        print(f"  Solving {label} ...")
        results[label] = solve_opf(
            mode,
            p_pv_mu=pv_mu,       p_wind_mu=wind_mu,
            p_pv_sigma=sigma_pv, p_wind_sigma=sigma_wind,
            p_pv_low=pv_low,     p_wind_low=wind_low,
            epsilon=opf["epsilon"],
        )
        r = results[label]
        print(f"    cost={r['cost']:.3f}  p_slack={r['p_slack']:.4f}  "
              f"time={r['time_s']:.2f}s  status={r['status']}")

    utils.plot_opf_cost(results)
    utils.plot_voltage_profile(results)

    # Step 3: Monte Carlo validation
    print("\n=== Step 3: Monte Carlo Validation ===")
    mc_stats, caps = run_mc(
        results,
        p_load_total=P_LOAD_TOTAL,
        pv_mu=pv_mu,         wind_mu=wind_mu,
        sigma_pv=sigma_pv,   sigma_wind=sigma_wind,
        n_scenarios=mc["n_scenarios"],
        n_hours=mc["n_hours"],
        k_volt=mc["k_volt"],
        cap_margin=mc["cap_margin"],
        seed=mc["seed"],
    )
    summarize(mc_stats)

    utils.plot_gap_rate(mc_stats)
    utils.plot_tradeoff(mc_stats)
    utils.plot_voltage_heatmap(mc_stats)
    utils.plot_cost_comparison(mc_stats)
    utils.plot_voltage_pdf(mc_stats)

    # Step 4: save results
    utils.save_metrics(results, mc_stats)
    print("\nDone! Results saved to results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Simple IEEE 34-bus OPF")
    parser.add_argument("--config", default="configs/simple_ieee34.yaml")
    parser.add_argument("--data",   default=None, help="Glob pattern for CSV files")
    parser.add_argument("--epochs", type=int, default=None)
    main(parser.parse_args())
