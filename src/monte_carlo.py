"""
src/monte_carlo.py
Monte Carlo operational validation — vectorised NumPy (no OPF calls)

Why no OPF calls in validation:
    The OPF power balance equation gives p_slack = P_LOAD_TOTAL - p_pv - p_wind,
    so the solver reduces to a subtraction. Vectorised NumPy runs 500 x 24
    scenarios in under 1 second vs. hours for repeated OPF calls, with identical results.

Cost model:
    hour_cost = plan_cost    if needed <= cap   (normal operation)
              = 100 * cap²   if needed > cap    (shortfall penalty)
"""

import numpy as np


def run_mc(
    results,
    p_load_total,
    pv_mu,
    wind_mu,
    sigma_pv,
    sigma_wind,
    n_scenarios=500,
    n_hours=24,
    k_volt=0.35,
    cap_margin=1.05,
    seed=42,
):
    """
    Run Monte Carlo operational validation for all OPF methods.

    Args:
        results       : dict of solve_opf outputs keyed by method name
        p_load_total  : Total system active load (MW)
        pv_mu         : PV expected output (MW)
        wind_mu       : Wind expected output (MW)
        sigma_pv      : PV standard deviation (MW)
        sigma_wind    : Wind standard deviation (MW)
        n_scenarios   : Number of Monte Carlo scenarios
        n_hours       : Hours per scenario
        k_volt        : Voltage drop per MW of shortfall (pu/MW)
        cap_margin    : Contracted supply cap = p_slack * cap_margin
        seed          : Random seed for reproducibility

    Returns:
        mc_stats : dict, one entry per method:
            'gap_rate'     : ndarray [n_scenarios]  fraction of hours with shortfall
            'v_viol_rate'  : ndarray [n_scenarios]  fraction of hours with voltage violation
            'actual_costs' : ndarray [n_scenarios]  mean hourly cost per scenario
            'voltage_mat'  : ndarray [n_scenarios, n_hours, n_buses]
            'plan_cost'    : float   planning-stage OPF cost
        caps : dict  contracted supply cap per method
    """
    methods = list(results.keys())
    v_lo, v_hi = 0.95, 1.05

    # Contracted supply caps
    caps = {m: results[m]["p_slack"] * cap_margin for m in methods}

    # Planning-stage base voltages per method
    base_voltages = {m: np.array(results[m]["voltages"]) for m in methods}

    # Shared random renewable output samples (same realisations for all methods)
    pv_max   = pv_mu   + 3.5 * sigma_pv
    wind_max = wind_mu + 3.5 * sigma_wind
    rng      = np.random.default_rng(seed)
    pv_mat   = np.clip(rng.normal(pv_mu,   sigma_pv,   (n_scenarios, n_hours)), 0.0, pv_max)
    wind_mat = np.clip(rng.normal(wind_mu, sigma_wind, (n_scenarios, n_hours)), 0.0, wind_max)
    needed   = np.maximum(0.0, p_load_total - pv_mat - wind_mat)   # [S, H]

    mc_stats = {}

    for method in methods:
        cap     = caps[method]
        base_vs = base_voltages[method]   # [n_buses]

        # Shortfall
        deficit  = np.maximum(0.0, needed - cap)   # [S, H]
        gap_bool = deficit > 0                      # [S, H]

        # Linearised voltage estimate  [S, H, n_buses]
        v_drop   = k_volt * deficit
        volt_mat = np.clip(
            base_vs[np.newaxis, np.newaxis, :] - v_drop[:, :, np.newaxis],
            0.88, 1.10,
        )

        # Voltage violation: any bus outside [0.95, 1.05]  [S, H]
        v_viol_bool = np.any((volt_mat < v_lo) | (volt_mat > v_hi), axis=2)

        # Cost: use plan_cost during normal operation, penalty during shortfall
        plan_cost_val = results[method]["cost"]
        penalty_cost  = 100.0 * cap ** 2
        hour_costs    = np.where(gap_bool, penalty_cost, plan_cost_val)

        mc_stats[method] = {
            "gap_rate"    : gap_bool.mean(axis=1),        # [S]
            "v_viol_rate" : v_viol_bool.mean(axis=1),     # [S]
            "actual_costs": hour_costs.mean(axis=1),      # [S]
            "voltage_mat" : volt_mat,                     # [S, H, buses]
            "plan_cost"   : plan_cost_val,
        }

    return mc_stats, caps


def summarize(mc_stats):
    """
    Print and return a summary table of MC validation results.

    Args:
        mc_stats : output of run_mc()

    Returns:
        df : pandas.DataFrame indexed by method name
    """
    import pandas as pd

    rows = []
    for m, s in mc_stats.items():
        rows.append({
            "Method"       : m,
            "Gap Rate"     : f"{s['gap_rate'].mean() * 100:.2f}%",
            "Scenario Gap" : f"{(s['gap_rate'] > 0).mean() * 100:.1f}%",
            "V-Viol Rate"  : f"{s['v_viol_rate'].mean() * 100:.1f}%",
            "Avg Cost ($)" : f"{s['actual_costs'].mean():.2f}",
            "Plan Cost ($)": f"{s['plan_cost']:.2f}",
        })
    df = pd.DataFrame(rows).set_index("Method")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70)
    return df
