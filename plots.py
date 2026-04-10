# plots.py
# Generate all visualizations from the experiment CSVs in results/.
#
# Usage:
#   python3 plots.py              # generate all plots
#   python3 plots.py convergence  # generate just one

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
BENCHMARK = 4.478


def load_csv(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  ⚠ {path} not found — run the experiment first")
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def ensure_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def save(fig, name):
    ensure_dir()
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


# -----------------------------------------------------------------------
# Plot 1: Benchmark comparison bar chart
# -----------------------------------------------------------------------
def plot_benchmark():
    print("\nPlot: Benchmark Comparison")
    data = load_csv("benchmark.csv")
    if not data: return

    methods = [r["method"] for r in data]
    prices  = [float(r["price"]) for r in data]
    times   = [float(r["runtime_s"]) for r in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # price bar chart
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    bars = ax1.bar(methods, prices, color=colors, edgecolor="black", linewidth=0.5)
    ax1.axhline(BENCHMARK, color="red", linestyle="--", linewidth=1, label=f"L&S benchmark={BENCHMARK}")
    ax1.set_ylabel("Estimated Price")
    ax1.set_title("American Put Price by Method")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=25)

    # runtime bar chart
    ax2.bar(methods, times, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Runtime (seconds)")
    ax2.set_title("Runtime by Method")
    ax2.tick_params(axis="x", rotation=25)

    fig.suptitle("LSMC Regression Methods — Benchmark Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "benchmark_comparison.png")


# -----------------------------------------------------------------------
# Plot 2: Convergence (price vs # paths)
# -----------------------------------------------------------------------
def plot_convergence():
    print("\nPlot: Convergence Analysis")
    data = load_csv("convergence.csv")
    if not data: return

    fig, ax = plt.subplots(figsize=(8, 5))
    methods_set = dict.fromkeys(r["method"] for r in data)  # ordered unique

    for method in methods_set:
        subset = [r for r in data if r["method"] == method]
        paths  = [int(r["paths"]) for r in subset]
        prices = [float(r["price"]) for r in subset]
        ax.plot(paths, prices, "o-", label=method, markersize=5)

    ax.axhline(BENCHMARK, color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"Benchmark={BENCHMARK}")
    ax.set_xlabel("Number of Paths")
    ax.set_ylabel("Estimated Price")
    ax.set_title("Convergence: Price vs Number of Paths")
    ax.legend()
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "convergence.png")


# -----------------------------------------------------------------------
# Plot 3: Hyperparameter sensitivity
# -----------------------------------------------------------------------
def plot_hyperparam():
    print("\nPlot: Hyperparameter Sensitivity")
    data = load_csv("hyperparam_sweep.csv")
    if not data: return

    # group by (method, param)
    groups = {}
    for r in data:
        key = (r["method"], r["param"])
        groups.setdefault(key, []).append(r)

    n = len(groups)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, ((method, param), subset) in enumerate(groups.items()):
        vals   = [float(r["value"]) for r in subset]
        prices = [float(r["price"]) for r in subset]
        axes[i].plot(vals, prices, "s-", color="steelblue", markersize=6)
        axes[i].axhline(BENCHMARK, color="red", linestyle="--", alpha=0.5)
        axes[i].set_xlabel(param)
        axes[i].set_ylabel("Price")
        axes[i].set_title(f"{method} — {param}")
        if max(vals) / (min(vals) + 1e-12) > 50:
            axes[i].set_xscale("log")
        axes[i].grid(True, alpha=0.3)

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Hyperparameter Sensitivity", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "hyperparam_sensitivity.png")


# -----------------------------------------------------------------------
# Plot 4: Runtime comparison
# -----------------------------------------------------------------------
def plot_runtime():
    print("\nPlot: Runtime Comparison")
    data = load_csv("runtime.csv")
    if not data: return

    methods = [r["method"] for r in data]
    prices  = [float(r["price"]) for r in data]
    times   = [float(r["avg_time_s"]) for r in data]
    errors  = [float(r["std_time_s"]) for r in data]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for i, (m, t, p) in enumerate(zip(methods, times, prices)):
        ax.scatter(t, abs(p - BENCHMARK), s=120, color=colors[i], edgecolors="black",
                   linewidth=0.5, zorder=3)
        ax.annotate(m, (t, abs(p - BENCHMARK)), textcoords="offset points",
                    xytext=(8, 4), fontsize=9)

    ax.set_xlabel("Avg Runtime (seconds)")
    ax.set_ylabel("|Price Error| vs Benchmark")
    ax.set_title("Accuracy vs Speed Tradeoff")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "runtime_tradeoff.png")


# -----------------------------------------------------------------------
# Plot 5: Option parameter variation heatmap
# -----------------------------------------------------------------------
def plot_option_params():
    print("\nPlot: Option Parameter Variation")
    data = load_csv("option_params.csv")
    if not data: return

    scenarios = list(dict.fromkeys(r["scenario"] for r in data))
    methods   = list(dict.fromkeys(r["method"] for r in data))

    # build matrix
    matrix = np.zeros((len(scenarios), len(methods)))
    for r in data:
        i = scenarios.index(r["scenario"])
        j = methods.index(r["method"])
        matrix[i, j] = float(r["price"])

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=9)

    # annotate cells
    for i in range(len(scenarios)):
        for j in range(len(methods)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, label="Price")
    ax.set_title("American Put Price Across Scenarios and Methods")
    fig.tight_layout()
    save(fig, "option_params_heatmap.png")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
PLOTS = {
    "benchmark": plot_benchmark,
    "convergence": plot_convergence,
    "hyperparam": plot_hyperparam,
    "runtime": plot_runtime,
    "optparams": plot_option_params,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "all":
        for fn in PLOTS.values():
            fn()
    elif sys.argv[1] in PLOTS:
        PLOTS[sys.argv[1]]()
    else:
        print(f"Unknown plot: {sys.argv[1]}")
        print(f"Available: {', '.join(PLOTS.keys())}, all")
        sys.exit(1)
