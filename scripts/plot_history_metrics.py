#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path


def _to_float(x: str) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    return float(s)


def load_metrics_csv(path: Path):
    iterations = []
    energy = []
    ang_momentum = []
    dcx = []
    dcy = []
    dcz = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "iteration",
            "energy",
            "ang_momentum",
            "density_center_x",
            "density_center_y",
            "density_center_z",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

        for row in reader:
            iterations.append(int(row["iteration"]))
            energy.append(_to_float(row["energy"]))
            ang_momentum.append(_to_float(row["ang_momentum"]))
            dcx.append(_to_float(row["density_center_x"]))
            dcy.append(_to_float(row["density_center_y"]))
            dcz.append(_to_float(row["density_center_z"]))

    return iterations, energy, ang_momentum, dcx, dcy, dcz


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot energy, angular momentum and density center vs iteration from SimulationHistory CSV export."
    )
    parser.add_argument("csv", type=Path, help="Input CSV file produced by SimulationHistory::saveMetricsToCSV")
    parser.add_argument(
        "--save-prefix",
        type=Path,
        default=None,
        help="If set, saves figures as '<prefix>_energy.png', '<prefix>_ang_momentum.png', '<prefix>_density_center.png'",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title prefix for plots",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(
            "matplotlib is required. Install with: pip install matplotlib\n" f"Original error: {e}"
        )

    it, E, L, cx, cy, cz = load_metrics_csv(args.csv)

    title_prefix = (args.title + " - ") if args.title else ""

    # Energy
    plt.figure()
    plt.plot(it, E, linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title(f"{title_prefix}Energy vs Iteration")
    plt.grid(True, alpha=0.3)
    if args.save_prefix:
        plt.savefig(str(args.save_prefix) + "_energy.png", dpi=160, bbox_inches="tight")

    # Angular momentum
    plt.figure()
    plt.plot(it, L, linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Angular momentum")
    plt.title(f"{title_prefix}Angular Momentum vs Iteration")
    plt.grid(True, alpha=0.3)
    if args.save_prefix:
        plt.savefig(str(args.save_prefix) + "_ang_momentum.png", dpi=160, bbox_inches="tight")

    # Density center (3 components)
    plt.figure()
    plt.plot(it, cx, label="cx", linewidth=1.5)
    plt.plot(it, cy, label="cy", linewidth=1.5)
    plt.plot(it, cz, label="cz", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Density center")
    plt.title(f"{title_prefix}Density Center vs Iteration")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if args.save_prefix:
        plt.savefig(str(args.save_prefix) + "_density_center.png", dpi=160, bbox_inches="tight")

    if not args.save_prefix:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
