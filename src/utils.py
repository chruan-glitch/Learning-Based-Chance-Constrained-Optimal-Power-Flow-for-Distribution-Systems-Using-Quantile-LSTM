"""
src/utils.py
Visualisation utilities + result saving.
Each plot is a separate function producing one figure — compatible with VS Code and Colab.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

COLORS  = {"Deterministic": "#2196F3", "CCOPF": "#4CAF50", "Robust": "#F44336"}
METHODS = ["Deterministic", "CCOPF", "Robust"]

FIG_DIR = os.path.join("results", "figures")
LOG_DIR = os.path.join("results", "logs")


def _savefig(fig, fname):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.show()
    plt.close(fig)


# ── OPF planning stage plots ───────────────────────────────────────────────

def plot_opf_cost(results, save=True):
    """Bar chart: planned operational cost for all three methods."""
    valid = {k: v for k, v in results.items() if v["cost"] is not None}
    names = [m for m in METHODS if m in valid]
    costs = [valid[n]["cost"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, costs, color=[COLORS[n] for n in names],
                  width=0.5, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Planned Operational Cost  (Det < CCOPF < Robust)", fontsize=12)
    ax.set_ylabel("Cost ($)")
    ax.set_ylim(0, max(costs) * 1.30)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        _savefig(fig, "opf_cost.png")


def plot_voltage_profile(results, save=True):
    """Line chart: node voltage profile for all three methods."""
    valid   = {k: v for k, v in results.items() if v["voltages"] is not None}
    names   = [m for m in METHODS if m in valid]
    n_buses = len(next(iter(valid.values()))["voltages"])
    buses   = list(range(1, n_buses + 1))
    markers = {"Deterministic": "o", "Robust": "s", "CCOPF": "^"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in names:
        ax.plot(buses, valid[name]["voltages"], label=name,
                marker=markers[name], markersize=3.5,
                linewidth=1.5, color=COLORS[name], alpha=0.9)
    ax.axhline(0.95, color="red", linestyle="--", linewidth=1.2)
    ax.axhline(1.05, color="red", linestyle="--", linewidth=1.2, label="Limits")
    ax.fill_between(buses, 0.95, 1.05, color="green", alpha=0.05, label="Safe zone")
    ax.set_title("Node Voltage Profile (Planning Stage)", fontsize=12)
    ax.set_xlabel("Bus Number")
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_xlim(1, n_buses)
    ax.set_ylim(0.88, 1.08)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        _savefig(fig, "voltage_profile.png")


# ── Monte Carlo plots ──────────────────────────────────────────────────────

def plot_gap_rate(mc_stats, save=True):
    """Bar chart (log scale): mean supply gap rate per method."""
    methods  = [m for m in METHODS if m in mc_stats]
    gap_vals = [mc_stats[m]["gap_rate"].mean() * 100 for m in methods]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(methods, gap_vals, color=[COLORS[m] for m in methods],
                  width=0.5, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, gap_vals):
        label_y = max(val, 0.05) * 2.5   # keep label above log-axis floor
        ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                f"{val:.2f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.03, 500)
    ax.set_yticks([0.05, 0.1, 0.5, 1, 5, 10, 50])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}%"))
    ax.axhline(5.0, color="black", linestyle="--", linewidth=1.2, label="ε=5% target")
    ax.set_title("Avg Supply Gap Rate (log scale)", fontsize=12)
    ax.set_ylabel("Gap Rate (%, log scale)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.4, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        _savefig(fig, "gap_rate.png")


def plot_tradeoff(mc_stats, save=True):
    """Scatter plot: economy–reliability trade-off across methods."""
    methods      = [m for m in METHODS if m in mc_stats]
    scenario_gap = [(mc_stats[m]["gap_rate"] > 0).mean() * 100 for m in methods]
    plan_c       = [mc_stats[m]["plan_cost"] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(methods):
        ax.scatter(scenario_gap[i], plan_c[i], s=400, color=COLORS[m],
                   zorder=5, edgecolors="white", linewidths=2)
        offsets = {"Deterministic": (6, -20), "CCOPF": (6, 6), "Robust": (-100, 6)}
        ox, oy = offsets[m]
        ax.annotate(f"{m}\n({scenario_gap[i]:.1f}%, ${plan_c[i]:.0f})",
                    xy=(scenario_gap[i], plan_c[i]),
                    xytext=(ox, oy), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=COLORS[m])
    pts = sorted(zip(scenario_gap, plan_c))
    for i in range(len(pts) - 1):
        ax.annotate("", xy=(pts[i][0], pts[i][1]),
                    xytext=(pts[i + 1][0], pts[i + 1][1]),
                    arrowprops=dict(arrowstyle="<-", color="gray", lw=1.5, linestyle="dashed"))
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlim(0.8, 200)
    ax.axvline(5.0, color="black", linestyle="--", linewidth=1.2, label="ε=5% target")
    ax.fill_betweenx([0, max(plan_c) * 1.5], 0.5, 5, color="green", alpha=0.07, label="Near-ideal")
    ax.set_xlabel("Scenario Gap Rate (%, log)  [lower = safer]", fontsize=10)
    ax.set_ylabel("Planning Cost ($)  [lower = cheaper]", fontsize=10)
    ax.set_title("Economy–Reliability Trade-off", fontsize=12)
    ax.set_ylim(0, max(plan_c) * 1.35)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        _savefig(fig, "tradeoff.png")


def plot_voltage_heatmap(mc_stats, method="CCOPF", scenario_idx=None, save=True):
    """Heatmap: node voltages over 24 hours for a sample scenario."""
    volt_mat = mc_stats[method]["voltage_mat"]
    if scenario_idx is None:
        scenario_idx = volt_mat.shape[0] // 2
    heat = volt_mat[scenario_idx]   # [n_hours, n_buses]

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(heat, aspect="auto", cmap="RdYlGn", vmin=0.93, vmax=1.07, origin="upper")
    plt.colorbar(im, ax=ax, label="|V| (p.u.)")
    ax.set_title(f"{method} Voltage Heatmap (Scenario {scenario_idx})", fontsize=12)
    ax.set_xlabel("Bus Number (1 → N)")
    ax.set_ylabel("Hour (0 → 23)")
    n_buses = heat.shape[1]
    ax.set_xticks(range(0, n_buses, 4))
    ax.set_xticklabels(range(1, n_buses + 1, 4))
    plt.tight_layout()
    if save:
        _savefig(fig, f"voltage_heatmap_{method.lower()}.png")


def plot_cost_comparison(mc_stats, save=True):
    """Grouped bar chart: planning cost vs mean actual cost per method."""
    methods  = [m for m in METHODS if m in mc_stats]
    plan_c   = [mc_stats[m]["plan_cost"] for m in methods]
    actual_c = [mc_stats[m]["actual_costs"].mean() for m in methods]

    fig, ax = plt.subplots(figsize=(7, 5))
    x, w = np.arange(len(methods)), 0.35
    b1 = ax.bar(x - w / 2, plan_c,   w, color=[COLORS[m] for m in methods],
                alpha=0.9, edgecolor="white", label="Planning Cost")
    b2 = ax.bar(x + w / 2, actual_c, w, color=[COLORS[m] for m in methods],
                alpha=0.5, hatch="xx", edgecolor="white", label="Avg Actual Cost")
    for bar, val in zip(list(b1) + list(b2), plan_c + actual_c):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                f"${val:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Planning vs Avg Actual Cost", fontsize=12)
    ax.set_ylabel("Cost ($)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        _savefig(fig, "cost_comparison.png")


def plot_voltage_pdf(mc_stats, save=True):
    """KDE plot: voltage probability density for all three methods."""
    try:
        import seaborn as sns
        has_sns = True
    except ImportError:
        has_sns = False

    methods = [m for m in METHODS if m in mc_stats]
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in methods:
        vmat = mc_stats[m]["voltage_mat"].flatten()
        vmat = vmat[~np.isnan(vmat)]
        vmat = vmat[(vmat >= 0.93) & (vmat <= 1.08)]
        if has_sns:
            sns.kdeplot(vmat, ax=ax, fill=True, alpha=0.25,
                        label=m, color=COLORS[m], linewidth=2)
        else:
            ax.hist(vmat, bins=60, density=True, alpha=0.3, label=m, color=COLORS[m])
    ax.axvline(0.95, color="red", linestyle="--", linewidth=1.5, label="Limits (0.95/1.05)")
    ax.axvline(1.05, color="red", linestyle="--", linewidth=1.5)
    ax.axvspan(0.95, 1.05, alpha=0.05, color="green", label="Safe zone")
    ax.set_xlabel("Voltage (p.u.)")
    ax.set_ylabel("Density")
    ax.set_title("Voltage PDF — All Three Methods", fontsize=12)
    ax.set_xlim(0.93, 1.08)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        _savefig(fig, "voltage_pdf.png")


def plot_quantile_predictions(model, X_test, y_test, scaler, target_indices,
                              num_samples=200, save=True):
    """Two-panel plot: quantile forecast vs actual for Solar and Wind."""
    import torch

    target_names = ["Solar (PV) Power", "Wind Power"]
    palette      = ["#1565C0", "#2E7D32"]

    model.eval()
    with torch.no_grad():
        y_pred_all = model(X_test[:num_samples]).numpy()
    y_true_all = y_test[:num_samples].numpy()

    def _inv(arr, t_idx):
        dummy = np.zeros((len(arr), len(scaler.scale_)))
        dummy[:, target_indices[t_idx]] = arr
        raw = scaler.inverse_transform(dummy)[:, target_indices[t_idx]] / 1000.0
        return np.maximum(raw, 0.0)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for t in range(2):
        true     = _inv(y_true_all[:, t], t)
        q10      = _inv(y_pred_all[:, t, 0], t)
        q50      = _inv(y_pred_all[:, t, 1], t)
        q90      = _inv(y_pred_all[:, t, 2], t)
        coverage = float(np.mean((true >= q10) & (true <= q90))) * 100

        ax = axes[t]
        ax.plot(true, color="black", linewidth=1.0, linestyle="--",
                alpha=0.8, label="Actual", zorder=3)
        ax.plot(q50, color=palette[t], linewidth=1.6, label="Median (q50)", zorder=4)
        ax.fill_between(range(num_samples), q10, q90, color=palette[t], alpha=0.20,
                        label=f"80% Interval [q10, q90]  (coverage={coverage:.1f}%)")
        ax.text(0.99, 0.97, f"Coverage: {coverage:.1f}%\n(Target: 80%)",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=palette[t],
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        ax.set_title(f"{target_names[t]} — Quantile Forecast vs Actual", fontsize=11)
        ax.set_ylabel("Power (MW)")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Time Steps (Hours)")
    plt.tight_layout()
    if save:
        _savefig(fig, "quantile_predictions.png")


# ── Result saving ──────────────────────────────────────────────────────────

def save_metrics(results, mc_stats, filepath=None):
    """
    Save OPF planning results and MC statistics to JSON.

    Args:
        results   : dict of solve_opf outputs
        mc_stats  : dict of run_mc outputs
        filepath  : output path (default: results/logs/metrics.json)
    """
    if filepath is None:
        os.makedirs(LOG_DIR, exist_ok=True)
        filepath = os.path.join(LOG_DIR, "metrics.json")

    out = {}
    for m, res in results.items():
        s = mc_stats.get(m, {})
        out[m] = {
            "plan_cost"  : res.get("cost"),
            "p_slack"    : res.get("p_slack"),
            "solver_time": res.get("time_s"),
            "status"     : res.get("status"),
            "mc_gap_rate": float(s["gap_rate"].mean() * 100)    if "gap_rate"     in s else None,
            "mc_vviol"   : float(s["v_viol_rate"].mean() * 100) if "v_viol_rate"  in s else None,
            "mc_avg_cost": float(s["actual_costs"].mean())       if "actual_costs" in s else None,
        }

    with open(filepath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Metrics saved to {filepath}")
