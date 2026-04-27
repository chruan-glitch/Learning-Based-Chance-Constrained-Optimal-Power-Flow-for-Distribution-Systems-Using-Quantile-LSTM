# Stochastic OPF — IEEE 34-bus

> **QR-LSTM Quantile Forecasting + DistFlow-SOCP Stochastic Optimal Power Flow**
>
> Comparing three planning strategies — **Deterministic / CCOPF / Robust** — on economy and supply reliability under renewable energy uncertainty.

---

## Background

Modern distribution networks with high penetration of solar and wind face significant output uncertainty. This project builds a complete pipeline:

1. **QR-LSTM** forecasts solar and wind power with uncertainty intervals (q10/q50/q90)
2. **DistFlow-SOCP OPF** (Pyomo + IPOPT) solves optimal power flow under three uncertainty handling strategies
3. **Monte Carlo validation** (500 scenarios × 24 hours) evaluates each strategy's real-world reliability

Two network cases are provided: a **simplified** uniform-parameter 34-bus and the **true** IEEE 34-bus with realistic impedances.

---

## Project Structure

```
stochastic-opf-ieee34/
├── src/
│   ├── __init__.py
│   ├── dataset.py          # Weather data loading, physical modelling, leak-free split
│   ├── model.py            # QRLSTM model + quantile loss
│   ├── train.py            # Training loop
│   ├── opf_simple.py       # Simplified 34-bus OPF (uniform parameters)
│   ├── opf_true.py         # True IEEE 34-bus OPF (realistic parameters)
│   ├── monte_carlo.py      # Vectorised Monte Carlo validation (no OPF calls)
│   └── utils.py            # Visualisation + result saving
├── configs/
│   ├── simple_ieee34.yaml  # Parameters for simplified case
│   └── true_ieee34.yaml    # Parameters for true IEEE 34-bus case
├── scripts/
│   ├── run_simple.py       # One-click runner: simplified case
│   └── run_true.py         # One-click runner: true case
├── notebooks/
│   ├── 01_Simple_IEEE34_Demo.ipynb
│   └── 02_True_IEEE34_Case.ipynb
├── data/                   # Place weather CSV files here (not committed)
├── results/
│   ├── figures/            # Auto-saved plots
│   └── logs/
│       └── metrics.json
├── docs/
│   └── report.md
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Install the IPOPT solver (required by Pyomo):

```bash
# Conda (recommended)
conda install -c conda-forge ipopt -y

# Google Colab
!conda install -c conda-forge ipopt -y
```

### 2. One-click run (no Jupyter needed)

```bash
# Simplified IEEE 34-bus — OPF + MC only (no LSTM training)
python scripts/run_simple.py

# True IEEE 34-bus — OPF + MC only
python scripts/run_true.py

# With QRLSTM training (requires weather data in data/)
python scripts/run_true.py --data "data/900131_*.csv" --epochs 50
```

### 3. Notebook walkthrough

Open `notebooks/` in Jupyter Lab / VS Code / Colab and run cells in order.
Each visualisation is a separate cell for compatibility with VS Code's output height limit.

---

## Method Comparison

| Method | Strategy | Gap Rate | Planning Cost |
|--------|----------|----------|---------------|
| **Deterministic** | Plan with expected renewable output | ~45% | Lowest |
| **CCOPF** | Chance-constraint (ε = 5%), balance cost & reliability | ~0.5% | Medium |
| **Robust** | Worst-case (3σ), most conservative | ~0% | Highest |

**Gap Rate**: fraction of operating hours where actual demand exceeds contracted supply cap.

---

## Network Cases

### Simplified IEEE 34-bus (`src/opf_simple.py`)
- 34 buses, chain topology, uniform impedance (r=0.01, x=0.02 pu)
- Uniform load: 0.05 MW per bus (total 1.70 MW)
- PV @ Bus 34, Wind @ Bus 20

### True IEEE 34-bus (`src/opf_true.py`)
- 34 buses, branching topology, 33 lines
- Base: 24.9 kV / 1 MVA (Z_base ≈ 620 Ω)
- Real impedances from line configurations (l1/l2/l3/lx) and lengths
- Non-uniform loads: distributed base + concentrated spot loads (total ≈ 1.072 MVA)
- PV @ Bus 20 (original Bus 834), Wind @ Bus 9 (original Bus 816)

**Bug fixes applied vs original notebooks:**
- `_R/_X` dict comprehension fixed (indexing instead of tuple unpacking)
- `Q_LOAD_IEEE` unified to pf = 0.9
- MC validation fully vectorised — removes erroneous 12,000 OPF calls

---

## Key Design Decisions

### Why no OPF calls in Monte Carlo?

The OPF power balance equation gives `p_slack = P_LOAD_TOTAL - p_pv - p_wind`, so the solver reduces to a subtraction. Vectorised NumPy runs 500 × 24 scenarios in under 1 second vs. hours for repeated OPF calls, with identical results.

### Cost model

```
hour_cost = plan_cost    if needed ≤ cap   (normal operation)
          = 100 × cap²   if needed > cap   (shortfall penalty)
```

### Robust uses 3σ (not 2σ)

With 2σ (z=2.0) and CCOPF at ε=0.05 (z=1.645), strategies are too close. Using 3σ for Robust gives clear separation: Det < CCOPF < Robust in both cost and reliability.

---

## Data Format

Weather CSV files (skip first 2 rows) with columns:
`Wind Speed`, `Temperature`, `Pressure`, `GHI`, `Hour`, `Month`

Place files in `data/` — excluded from version control via `.gitignore`.

---

## Results Output

```
results/figures/    opf_cost.png, voltage_profile.png, gap_rate.png,
                    tradeoff.png, voltage_heatmap_ccopf.png,
                    cost_comparison.png, voltage_pdf.png,
                    quantile_predictions.png, epsilon_tradeoff.png
results/logs/       metrics.json
```
