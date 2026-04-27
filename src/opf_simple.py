"""
src/opf_simple.py
DistFlow-SOCP OPF — Simplified IEEE 34-bus (uniform parameters, chain topology)
Corresponds to notebooks/01_Simple_IEEE34_Demo.ipynb
"""

import time
import numpy as np
import pyomo.environ as pyo
from scipy.stats import norm

# -- Network parameters (simplified uniform)
N_BUSES      = 34
NODES        = list(range(1, N_BUSES + 1))
LINES        = [(i, i + 1) for i in range(1, N_BUSES)]
R            = {lk: 0.01 for lk in LINES}   # resistance (pu)
X            = {lk: 0.02 for lk in LINES}   # reactance  (pu)
P_LOAD       = {i: 0.05  for i in NODES}    # active load per bus (MW)
Q_LOAD       = {i: 0.02  for i in NODES}    # reactive load per bus (MVAR)
P_LOAD_TOTAL = sum(P_LOAD.values())          # 1.70 MW

SLACK_BUS = 1
PV_BUS    = 34
WIND_BUS  = 20

V_LO, V_HI = 0.95, 1.05


def solve_opf(
    mode,
    p_pv_mu=0.0,
    p_wind_mu=0.0,
    p_pv_sigma=0.0,
    p_wind_sigma=0.0,
    p_pv_low=None,
    p_pv_high=None,
    p_wind_low=None,
    p_wind_high=None,
    epsilon=0.05,
):
    """
    DistFlow-SOCP OPF — Simplified IEEE 34-bus (Pyomo + IPOPT)

    Args:
        mode          : 'det' | 'robust' | 'ccopf'
        p_pv_mu       : PV expected output (MW)
        p_wind_mu     : Wind expected output (MW)
        p_pv_sigma    : PV standard deviation — used by CCOPF
        p_wind_sigma  : Wind standard deviation — used by CCOPF
        p_pv_low      : PV worst-case output — used by Robust (None → mu × 0.8)
        p_wind_low    : Wind worst-case output — used by Robust (None → mu × 0.8)
        epsilon       : Allowed violation probability — used by CCOPF (default 0.05)

    Returns:
        dict with keys: cost (float|None), voltages (list[float]|None),
                        p_slack (float|None), status (str), time_s (float)
    """
    tan_phi      = np.tan(np.arccos(0.95))
    V_LO2, V_HI2 = V_LO ** 2, V_HI ** 2

    # -- Effective renewable output per mode
    if mode == "det":
        ren_pv, ren_wnd = p_pv_mu, p_wind_mu
    elif mode == "robust":
        ren_pv  = p_pv_low   if p_pv_low   is not None else p_pv_mu  * 0.8
        ren_wnd = p_wind_low if p_wind_low is not None else p_wind_mu * 0.8
    elif mode == "ccopf":
        z       = float(norm.ppf(1 - epsilon))
        ren_pv  = max(0.0, p_pv_mu  - z * p_pv_sigma)
        ren_wnd = max(0.0, p_wind_mu - z * p_wind_sigma)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'det', 'robust', or 'ccopf'.")

    # -- Build Pyomo model
    mdl = pyo.ConcreteModel()
    mdl.N = pyo.Set(initialize=NODES)
    mdl.L = pyo.Set(initialize=LINES, dimen=2)

    mdl.v       = pyo.Var(mdl.N, bounds=(V_LO2, V_HI2), initialize=1.0)
    mdl.P       = pyo.Var(mdl.L, initialize=0.0)
    mdl.Q       = pyo.Var(mdl.L, initialize=0.0)
    mdl.l       = pyo.Var(mdl.L, bounds=(0, None), initialize=0.0)
    mdl.p_slack = pyo.Var(bounds=(-10.0, 10.0), initialize=1.0)
    mdl.q_slack = pyo.Var(bounds=(-10.0, 10.0), initialize=0.0)
    mdl.v[SLACK_BUS].fix(1.0)

    @mdl.Constraint(mdl.L)
    def soc_con(m, i, j):
        return m.P[i, j] ** 2 + m.Q[i, j] ** 2 <= m.v[i] * m.l[i, j]

    @mdl.Constraint(mdl.L)
    def v_drop(m, i, j):
        return m.v[j] == (
            m.v[i]
            - 2 * (R[i, j] * m.P[i, j] + X[i, j] * m.Q[i, j])
            + (R[i, j] ** 2 + X[i, j] ** 2) * m.l[i, j]
        )

    @mdl.Constraint(mdl.N)
    def p_balance(m, i):
        in_p  = sum(m.P[k] for k in LINES if k[1] == i)
        out_p = sum(m.P[k] for k in LINES if k[0] == i)
        gen   = m.p_slack if i == SLACK_BUS else 0.0
        ren   = (ren_pv if i == PV_BUS else ren_wnd if i == WIND_BUS else 0.0)
        return in_p - out_p + gen + ren == P_LOAD[i]

    @mdl.Constraint(mdl.N)
    def q_balance(m, i):
        in_q  = sum(m.Q[k] for k in LINES if k[1] == i)
        out_q = sum(m.Q[k] for k in LINES if k[0] == i)
        gen_q = m.q_slack if i == SLACK_BUS else 0.0
        ren_q = (
            ren_pv  * tan_phi if i == PV_BUS  else
            ren_wnd * tan_phi if i == WIND_BUS else 0.0
        )
        return in_q - out_q + gen_q + ren_q == Q_LOAD[i]

    mdl.obj = pyo.Objective(expr=100.0 * mdl.p_slack ** 2, sense=pyo.minimize)

    # -- Solve
    t0     = time.time()
    solver = pyo.SolverFactory("ipopt")
    solver.options["max_iter"] = 3000
    solver.options["tol"]      = 1e-6
    res     = solver.solve(mdl, tee=False)
    elapsed = time.time() - t0

    tc = res.solver.termination_condition
    if tc != pyo.TerminationCondition.optimal:
        return {
            "cost": None, "voltages": None,
            "p_slack": None, "status": str(tc), "time_s": elapsed,
        }

    voltages = [float(np.sqrt(max(0.0, pyo.value(mdl.v[i])))) for i in NODES]
    return {
        "cost"    : float(pyo.value(mdl.obj)),
        "voltages": voltages,
        "p_slack" : float(pyo.value(mdl.p_slack)),
        "status"  : "optimal",
        "time_s"  : elapsed,
    }
