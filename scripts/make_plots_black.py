#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = "figures_black"

def ensure_outdir():
    os.makedirs(OUTDIR, exist_ok=True)

def apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor": "black",
        "axes.facecolor": "black",
        "savefig.facecolor": "black",
        "savefig.edgecolor": "black",
        "axes.edgecolor": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "grid.color": "white",
        "grid.alpha": 0.18,
        "axes.grid": True,
        "grid.linestyle": "-",
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.frameon": True,
        "legend.facecolor": "black",
        "legend.edgecolor": "white",
        "legend.framealpha": 0.25,
        "lines.linewidth": 2.2,
        "lines.markersize": 7.5,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 11,
    })

def style_spines(ax):
    for s in ax.spines.values():
        s.set_color("white")
        s.set_linewidth(1.2)

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def annotate_bar_values(ax, bars, fmt="{:.1f}", y_offset_frac=0.02):
    y_max = max(b.get_height() for b in bars)
    off = y_offset_frac * (y_max if y_max > 0 else 1.0)
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h + off,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=10,
            color="white",
            alpha=0.95,
        )

def plot_cpu_progression_fps_gflops():
    impl = ["cpu+naive", "cpu+optim", "cpu+simd", "cpu+omp (12 th)"]
    time_ms = np.array([3445.84, 1450.26, 253.171, 53.548], dtype=float)
    fps = np.array([5.804, 13.791, 78.998, 373.497], dtype=float)
    gflops = np.array([6.9, 16.4, 94.2, 445.2], dtype=float)
    speedup = np.array([1.00, 2.38, 13.61, 64.34], dtype=float)

    x = np.arange(len(impl))

    # FPS bar
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    style_spines(ax)
    bars = ax.bar(x, fps, alpha=0.92)
    ax.set_title("CPU optimization progression (N=8000, I=20, galaxy) — FPS", pad=14)
    ax.set_ylabel("FPS")
    ax.set_xticks(x)
    ax.set_xticklabels(impl, rotation=15, ha="right")
    ax.grid(True, axis="y")
    annotate_bar_values(ax, bars, fmt="{:.3g}")
    for i, s in enumerate(speedup):
        ax.text(i, fps[i] * 0.60, f"{s:.2f}×", ha="center", va="center", fontsize=11, alpha=0.95)
    save_fig(os.path.join(OUTDIR, "plot1_cpu_progression_fps.png"))

    # GFLOPS bar (log helps readability with large spread)
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    style_spines(ax)
    bars = ax.bar(x, gflops, alpha=0.92)
    ax.set_title("CPU optimization progression (N=8000, I=20, galaxy) — GFLOP/s", pad=14)
    ax.set_ylabel("GFLOP/s")
    ax.set_xticks(x)
    ax.set_xticklabels(impl, rotation=15, ha="right")
    ax.grid(True, axis="y")
    if (gflops.max() / max(gflops.min(), 1e-12)) > 30:
        ax.set_yscale("log")
        ax.set_ylabel("GFLOP/s (log scale)")
    annotate_bar_values(ax, bars, fmt="{:.3g}", y_offset_frac=0.04)
    save_fig(os.path.join(OUTDIR, "plot1_cpu_progression_gflops.png"))

def plot_omp_threads():
    threads = np.array([1, 2, 4, 6, 8, 12], dtype=int)
    time_ms = np.array([15081.2, 19738.4, 9935.38, 6520.51, 5042.09, 4222.8], dtype=float)
    fps = np.array([6.631, 5.066, 10.065, 15.336, 19.833, 23.681], dtype=float)
    gflops = np.array([111.2, 84.9, 168.7, 257.1, 332.5, 397.0], dtype=float)

    # FPS vs threads
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    style_spines(ax)
    ax.plot(threads, fps, marker="o", alpha=0.95, label="FPS")
    ax.set_title("OpenMP scaling (N=30000, I=100, galaxy) — FPS vs threads", pad=14)
    ax.set_xlabel("Threads")
    ax.set_ylabel("FPS")
    ax.set_xticks(threads)
    ax.grid(True)
    ax.legend(loc="best")
    ax.annotate(
        "2 threads anomaly\n(run-to-run noise / affinity / turbo)",
        xy=(2, fps[1]),
        xytext=(3.2, fps.max() * 0.55),
        arrowprops=dict(arrowstyle="->", lw=1.0, color="white", alpha=0.8),
        fontsize=10,
        alpha=0.9,
    )
    save_fig(os.path.join(OUTDIR, "plot2_omp_threads_fps.png"))

    # Speedup vs threads
    speedup = fps / fps[0]
    ideal = threads / threads[0]

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    style_spines(ax)
    ax.plot(threads, speedup, marker="o", alpha=0.95, label="Measured speedup")
    ax.plot(threads, ideal, marker=None, linestyle="--", alpha=0.55, label="Ideal (linear)")
    ax.set_title("OpenMP scaling (N=30000, I=100, galaxy) — speedup", pad=14)
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup (vs 1 thread)")
    ax.set_xticks(threads)
    ax.grid(True)
    ax.legend(loc="best")
    save_fig(os.path.join(OUTDIR, "plot2_omp_threads_speedup.png"))

def plot_omp_N_scaling():
    N = np.array([4000, 8000, 12000, 16000, 20000, 24000, 30000], dtype=int)
    time_ms = np.array([42.561, 160.053, 348.868, 623.311, 972.780, 1601.44, 2195.25], dtype=float)
    fps = np.array([1409.74, 374.876, 171.985, 96.260, 61.679, 37.466, 27.332], dtype=float)
    gflops = np.array([420.1, 446.9, 461.3, 459.0, 459.5, 402.0, 458.2], dtype=float)

    # FPS vs N
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    style_spines(ax)
    ax.plot(N, fps, marker="o", alpha=0.95)
    ax.set_title("OpenMP (12 threads, I=60, galaxy) — FPS vs N", pad=14)
    ax.set_xlabel("Number of bodies (N)")
    ax.set_ylabel("FPS")
    ax.grid(True)
    save_fig(os.path.join(OUTDIR, "plot3_omp_N_fps.png"))

    # GFLOPS vs N
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    style_spines(ax)
    ax.plot(N, gflops, marker="o", alpha=0.95)
    ax.set_title("OpenMP (12 threads, I=60, galaxy) — GFLOP/s vs N", pad=14)
    ax.set_xlabel("Number of bodies (N)")
    ax.set_ylabel("GFLOP/s")
    ax.grid(True)
    save_fig(os.path.join(OUTDIR, "plot3_omp_N_gflops.png"))

def plot_hetero_fraction_log():
    frac = np.array([0.00, 0.25, 0.50, 0.60, 0.75, 1.00], dtype=float)
    time_ms = np.array([118566, 89066.1, 59498.6, 47610.2, 29810.6, 205.376], dtype=float)
    fps = np.array([0.506, 0.674, 1.008, 1.260, 2.013, 292.147], dtype=float)
    gflops = np.array([8.5, 11.3, 16.9, 21.1, 33.7, 4897.5], dtype=float)

    # FPS vs fraction (log y)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    style_spines(ax)
    ax.plot(frac, fps, marker="o", alpha=0.95)
    ax.set_title("Heterogeneous offload (N=30000, I=60, galaxy) — FPS vs GPU fraction", pad=14)
    ax.set_xlabel("MURB_HETERO_GPU_FRACTION")
    ax.set_ylabel("FPS (log scale)")
    ax.set_yscale("log")
    ax.set_xticks(frac)
    ax.grid(True, which="both")
    save_fig(os.path.join(OUTDIR, "plot5_hetero_fraction_fps_log.png"))

    # GFLOPS vs fraction (log y)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    style_spines(ax)
    ax.plot(frac, gflops, marker="o", alpha=0.95)
    ax.set_title("Heterogeneous offload (N=30000, I=60, galaxy) — GFLOP/s vs GPU fraction", pad=14)
    ax.set_xlabel("MURB_HETERO_GPU_FRACTION")
    ax.set_ylabel("GFLOP/s (log scale)")
    ax.set_yscale("log")
    ax.set_xticks(frac)
    ax.grid(True, which="both")
    save_fig(os.path.join(OUTDIR, "plot5_hetero_fraction_gflops_log.png"))

def main():
    ensure_outdir()
    apply_dark_style()
    plot_cpu_progression_fps_gflops()
    plot_omp_threads()
    plot_omp_N_scaling()
    plot_hetero_fraction_log()
    print(f"[OK] Saved plots into: {OUTDIR}/")

if __name__ == "__main__":
    main()
