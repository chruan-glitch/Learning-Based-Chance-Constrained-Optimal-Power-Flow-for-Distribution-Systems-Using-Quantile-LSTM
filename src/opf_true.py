"""
src/opf_true.py
DistFlow-SOCP OPF — True IEEE 34-bus parameters (Pyomo + IPOPT)
Corresponds to notebooks/02_True_IEEE34_Case.ipynb

Bug fixes vs original notebooks:
  - _R/_X dict comprehension: use index access instead of tuple unpacking
    (original `for ro,_ in [_CFG[c]]` assigned the whole tuple to ro)
  - Q_LOAD_IEEE: unified to pf = 0.9 for all buses
    (original mixed 0.004 base with 0.484 spot-load coefficient)
"""

import time
import numpy as np
import pyomo.environ as pyo
from scipy.stats import norm

# -- System base values
V_BASE_KV  = 24.9       # kV
S_BASE_MVA = 1.0        # MVA
Z_BASE     = (V_BASE_KV * 1e3) ** 2 / (S_BASE_MVA * 1e6)   # ≈ 620.01 Ω

# -- Line configuration impedances (Ω/mile)
_CFG = {
    "l1": (1.3368, 1.3343),   # single-phase #4/0 ACSR
    "l2": (0.7526, 1.1779),   # three-phase
    "l3": (1.3238, 1.3569),   # two-phase
    "lx": (0.0640, 0.1921),   # transformer equivalent
}

# -- Raw line data (from_bus, to_bus, length_miles, config)
_RAW = [
    (1,  2,  2.580, "l1"), (2,  3,  1.730, "l1"), (3,  4, 32.230, "l2"),
    (4,  5,  5.804, "l3"), (4,  6, 37.500, "l2"), (6,  7, 29.730, "l2"),
    (7,  8,  0.010, "l2"), (8,  9,  0.310, "l2"), (9, 10,  1.710, "l3"),
    (10, 11, 48.150, "l3"), (11, 12, 13.740, "l3"),
    (9,  13, 10.210, "l2"), (13, 14,  3.030, "l3"), (13, 15,  0.840, "l2"),
    (15, 16, 20.440, "l2"), (16, 17,  0.010, "l2"), (17, 33, 23.330, "l2"),
    (17, 34, 36.830, "l2"), (34, 18,  0.010, "l2"), (18, 19,  4.900, "l2"),
    (18, 31,  0.010, "lx"), (31, 32, 10.560, "lx"), (19, 20,  5.090, "l2"),
    (19, 30,  1.620, "l3"), (20, 21,  0.280, "l2"), (21, 22,  1.350, "l2"),
    (22, 23,  3.640, "l2"), (23, 24,  0.530, "l2"), (20, 25,  2.020, "l2"),
    (25, 26,  2.680, "l2"), (26, 27,  0.860, "l2"), (26, 28,  0.280, "l2"),
    (28, 29,  4.860, "l3"),
]

# -- Per-unit line impedances (fixed: direct index access, no unpacking bug)
_R     = {(f, t): _CFG[c][0] * l / Z_BASE for f, t, l, c in _RAW}
_X     = {(f, t): _CFG[c][1] * l / Z_BASE for f, t, l, c in _RAW}
TREE_L = [(f, t) for f, t, *_ in _RAW]   # 33 directed edges

# -- Bus active loads (MVA): distributed base + concentrated spot loads
_SPOT_P = {5: 0.040, 11: 0.040, 12: 0.040, 20: 0.126,
           22: 0.135, 23: 0.135, 24: 0.135, 32: 0.115}
P_LOAD_IEEE = {i: 0.009 + _SPOT_P.get(i, 0.0) for i in range(1, 35)}

# -- Bus reactive loads (fixed: unified pf = 0.9 across all buses)
_TAN_PHI    = np.tan(np.arccos(0.9))
Q_LOAD_IEEE = {i: P_LOAD_IEEE[i] * _TAN_PHI for i in range(1, 35)}

P_LOAD_TOTAL_IEEE = sum(P_LOAD_IEEE.values())   # ≈ 1.072 MVA
P_LOAD_TOTAL      = P_LOAD_TOTAL_IEEE            # alias for external use

# -- Renewable energy connection buses
SLACK_BUS = 1
PV_BUS    = 20   # original Bus 834
WIND_BUS  = 9    # original Bus 816

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
    DistFlow-SOCP OPF — True IEEE 34-bus (Pyomo + IPOPT)

    Args:
        mode          : 'det' | 'robust' | 'ccopf'
        p_pv_mu       : PV expected output (MVA)
        p_wind_mu     : Wind expected output (MVA)
        p_pv_sigma    : PV standard deviation — used by CCOPF
        p_wind_sigma  : Wind standard deviation — used by CCOPF
        p_pv_low      : PV worst-case output — used by Robust (None → mu × 0.8)
        p_wind_low    : Wind worst-case output — used by Robust (None → mu × 0.8)
        epsilon       : Allowed violation probability — used by CCOPF (default 0.05)

    Returns:
        dict with keys: cost (float|None), voltages (list[float]|None),
                        p_slack (float|None), status (str), time_s (float)
    """
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
    N   = list(range(1, 35))
    mdl = pyo.ConcreteModel()
    mdl.N = pyo.Set(initialize=N)
    mdl.L = pyo.Set(initialize=TREE_L, dimen=2)

    mdl.v       = pyo.Var(mdl.N, bounds=(V_LO2, V_HI2), initialize=1.0)
    mdl.P       = pyo.Var(mdl.L, initialize=0.0)
    mdl.Q       = pyo.Var(mdl.L, initialize=0.0)
    mdl.l       = pyo.Var(mdl.L, bounds=(0, None), initialize=0.0)
    mdl.p_slack = pyo.Var(bounds=(-5.0, 5.0), initialize=0.5)
    mdl.q_slack = pyo.Var(bounds=(-5.0, 5.0), initialize=0.0)
    mdl.v[SLACK_BUS].fix(1.0)

    @mdl.Constraint(mdl.L)
    def soc_con(m, i, j):
        return m.P[i, j] ** 2 + m.Q[i, j] ** 2 <= m.v[i] * m.l[i, j]

    @mdl.Constraint(mdl.L)
    def v_drop(m, i, j):
        return m.v[j] == (
            m.v[i]
            - 2 * (_R[i, j] * m.P[i, j] + _X[i, j] * m.Q[i, j])
            + (_R[i, j] ** 2 + _X[i, j] ** 2) * m.l[i, j]
        )

    @mdl.Constraint(mdl.N)
    def p_balance(m, i):
        in_p  = sum(m.P[k] for k in TREE_L if k[1] == i)
        out_p = sum(m.P[k] for k in TREE_L if k[0] == i)
        gen   = m.p_slack if i == SLACK_BUS else 0.0
        ren   = (ren_pv if i == PV_BUS else ren_wnd if i == WIND_BUS else 0.0)
        return in_p - out_p + gen + ren == P_LOAD_IEEE[i]

    @mdl.Constraint(mdl.N)
    def q_balance(m, i):
        in_q  = sum(m.Q[k] for k in TREE_L if k[1] == i)
        out_q = sum(m.Q[k] for k in TREE_L if k[0] == i)
        gen_q = m.q_slack if i == SLACK_BUS else 0.0
        ren_q = (
            ren_pv  * _TAN_PHI if i == PV_BUS  else
            ren_wnd * _TAN_PHI if i == WIND_BUS else 0.0
        )
        return in_q - out_q + gen_q + ren_q == Q_LOAD_IEEE[i]

    mdl.obj = pyo.Objective(expr=100.0 * mdl.p_slack ** 2, sense=pyo.minimize)

    # -- Solve
    t0     = time.time()
    solver = pyo.SolverFactory("ipopt")
    solver.options["max_iter"] = 5000
    solver.options["tol"]      = 1e-6
    res     = solver.solve(mdl, tee=False)
    elapsed = time.time() - t0

    tc = res.solver.termination_condition
    if tc != pyo.TerminationCondition.optimal:
        return {
            "cost": None, "voltages": None,
            "p_slack": None, "status": str(tc), "time_s": elapsed,
        }

    voltages = [float(np.sqrt(max(0.0, pyo.value(mdl.v[i])))) for i in N]
    return {
        "cost"    : float(pyo.value(mdl.obj)),
        "voltages": voltages,
        "p_slack" : float(pyo.value(mdl.p_slack)),
        "status"  : "optimal",
        "time_s"  : elapsed,
    }
