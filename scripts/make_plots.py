import os
import math
import matplotlib.pyplot as plt

def _apply_style():
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def _annotate_bars(ax, bars, labels, yscale="linear"):
    for bar, lab in zip(bars, labels):
        h = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2.0
        if yscale == "log":
            y = h * 1.15
        else:
            y = h + (0.02 * ax.get_ylim()[1])
        ax.text(x, y, lab, ha="center", va="bottom")

def plot1_cpu_progression(fps_log_gflops=True):
    impl = ["cpu+naive", "cpu+optim", "cpu+simd", "cpu+omp (12 th)"]
    time_ms = [3445.84, 1450.26, 253.171, 53.548]
    fps = [5.804, 13.791, 78.998, 373.497]
    gflops = [6.9, 16.4, 94.2, 445.2]
    speedup = ["1.00x", "2.38x", "13.61x", "64.34x"]

    os.makedirs("figures", exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(impl, fps)
    ax.set_title("CPU progression (N=8000, I=20, galaxy, fp32, --nv)")
    ax.set_ylabel("FPS")
    ax.set_xlabel("Implementation")
    ax.set_ylim(0, max(fps) * 1.28)
    ax.tick_params(axis="x", rotation=12)
    _annotate_bars(ax, bars, speedup, yscale="linear")
    fig.tight_layout()
    fig.savefig("figures/plot1_cpu_progression_fps.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(impl, gflops)
    ax.set_title("CPU progression (N=8000, I=20, galaxy, fp32, --nv)")
    ax.set_ylabel("GFLOP/s")
    ax.set_xlabel("Implementation")
    ax.tick_params(axis="x", rotation=12)

    if fps_log_gflops:
        ax.set_yscale("log")
        ax.set_ylim(min(gflops) / 1.6, max(gflops) * 2.2)
        value_labels = [f"{v:.1f}\n({s})" for v, s in zip(gflops, speedup)]
        _annotate_bars(ax, bars, value_labels, yscale="log")
    else:
        ax.set_ylim(0, max(gflops) * 1.28)
        value_labels = [f"{v:.1f} ({s})" for v, s in zip(gflops, speedup)]
        _annotate_bars(ax, bars, value_labels, yscale="linear")

    fig.tight_layout()
    fig.savefig("figures/plot1_cpu_progression_gflops.png")
    plt.close(fig)

def plot2_omp_threads():
    threads = [1, 2, 4, 6, 8, 12]
    time_ms = [15081.2, 19738.4, 9935.38, 6520.51, 5042.09, 4222.8]
    fps = [6.631, 5.066, 10.065, 15.336, 19.833, 23.681]
    gflops = [111.2, 84.9, 168.7, 257.1, 332.5, 397.0]

    os.makedirs("figures", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(threads, fps, marker="o", linewidth=2)
    ax.set_title("OpenMP scaling vs threads (N=30000, I=100, galaxy, fp32, --nv)")
    ax.set_xlabel("Threads")
    ax.set_ylabel("FPS")
    ax.set_xticks(threads)
    ax.set_ylim(0, max(fps) * 1.2)
    for t, v in zip(threads, fps):
        ax.text(t, v + 0.02 * max(fps), f"{v:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig("figures/plot2_omp_threads_fps.png")
    plt.close(fig)

    base = fps[0]
    speedup = [v / base for v in fps]
    ideal = threads

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(threads, speedup, marker="o", linewidth=2, label="Measured speedup")
    ax.plot(threads, ideal, linestyle="--", linewidth=2, label="Ideal (linear)")
    ax.set_title("OpenMP speedup vs threads (baseline = 1 thread)")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup")
    ax.set_xticks(threads)
    ax.set_ylim(0, max(max(speedup), max(ideal)) * 1.15)
    ax.legend(loc="best")
    for t, v in zip(threads, speedup):
        ax.text(t, v + 0.02 * max(max(speedup), max(ideal)), f"{v:.2f}x", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig("figures/plot2_omp_threads_speedup.png")
    plt.close(fig)

def plot3_omp_vs_N():
    N = [4000, 8000, 12000, 16000, 20000, 24000, 30000]
    time_ms = [42.561, 160.053, 348.868, 623.311, 972.780, 1601.44, 2195.25]
    fps = [1409.74, 374.876, 171.985, 96.260, 61.679, 37.466, 27.332]
    gflops = [420.1, 446.9, 461.3, 459.0, 459.5, 402.0, 458.2]

    os.makedirs("figures", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(N, fps, marker="o", linewidth=2)
    ax.set_title("OpenMP (12 threads): FPS vs N (I=60, galaxy, fp32, --nv)")
    ax.set_xlabel("Number of bodies (N)")
    ax.set_ylabel("FPS")
    ax.set_xticks(N)
    ax.tick_params(axis="x", rotation=0)
    ax.set_ylim(0, max(fps) * 1.15)
    for x, v in zip(N, fps):
        ax.text(x, v + 0.02 * max(fps), f"{v:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig("figures/plot3_omp_N_fps.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(N, gflops, marker="o", linewidth=2)
    ax.set_title("OpenMP (12 threads): GFLOP/s vs N (I=60, galaxy, fp32, --nv)")
    ax.set_xlabel("Number of bodies (N)")
    ax.set_ylabel("GFLOP/s")
    ax.set_xticks(N)
    ax.set_ylim(min(gflops) * 0.85, max(gflops) * 1.12)
    for x, v in zip(N, gflops):
        ax.text(x, v + 0.01 * max(gflops), f"{v:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig("figures/plot3_omp_N_gflops.png")
    plt.close(fig)

def plot5_hetero_fraction_log():
    frac = [0.00, 0.25, 0.50, 0.60, 0.75, 1.00]
    time_ms = [118566, 89066.1, 59498.6, 47610.2, 29810.6, 205.376]
    fps = [0.506, 0.674, 1.008, 1.260, 2.013, 292.147]
    gflops = [8.5, 11.3, 16.9, 21.1, 33.7, 4897.5]

    os.makedirs("figures", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(frac, fps, marker="o", linewidth=2)
    ax.set_yscale("log")
    ax.set_title("Heterogeneous: FPS vs GPU fraction (N=30000, I=60, galaxy, fp32, --nv)")
    ax.set_xlabel("MURB_HETERO_GPU_FRACTION")
    ax.set_ylabel("FPS (log scale)")
    ax.set_xticks(frac)
    ax.set_ylim(min(fps) / 1.8, max(fps) * 2.0)
    for x, v in zip(frac, fps):
        ax.text(x, v * 1.15, f"{v:.3g}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig("figures/plot5_hetero_fraction_fps_log.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(frac, gflops, marker="o", linewidth=2)
    ax.set_yscale("log")
    ax.set_title("Heterogeneous: GFLOP/s vs GPU fraction (N=30000, I=60, galaxy, fp32, --nv)")
    ax.set_xlabel("MURB_HETERO_GPU_FRACTION")
    ax.set_ylabel("GFLOP/s (log scale)")
    ax.set_xticks(frac)
    ax.set_ylim(min(gflops) / 1.8, max(gflops) * 2.0)
    for x, v in zip(frac, gflops):
        ax.text(x, v * 1.15, f"{v:.3g}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig("figures/plot5_hetero_fraction_gflops_log.png")
    plt.close(fig)

def main():
    _apply_style()
    plot1_cpu_progression(fps_log_gflops=True)
    plot2_omp_threads()
    plot3_omp_vs_N()
    plot5_hetero_fraction_log()
    print("Done. Figures saved in ./figures/")
    print("Generated:")
    print(" - figures/plot1_cpu_progression_fps.png")
    print(" - figures/plot1_cpu_progression_gflops.png")
    print(" - figures/plot2_omp_threads_fps.png")
    print(" - figures/plot2_omp_threads_speedup.png")
    print(" - figures/plot3_omp_N_fps.png")
    print(" - figures/plot3_omp_N_gflops.png")
    print(" - figures/plot5_hetero_fraction_fps_log.png")
    print(" - figures/plot5_hetero_fraction_gflops_log.png")

if __name__ == "__main__":
    main()
