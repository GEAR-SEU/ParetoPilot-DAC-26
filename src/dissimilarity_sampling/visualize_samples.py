import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config_loader import get_config_value


WLUT = 0.3
WFF = 0.25
WDSP = 0.3
WBRAM = 0.05


@dataclass
class ParetoVisualizer:
    pareto_full_dir: str = field(default_factory=lambda: get_config_value("visualization", "pareto_full_dir", default="DSE_Data"))
    pareto_samples_dir: str = field(default_factory=lambda: get_config_value("visualization", "pareto_samples_dir", default="DSE_Data/optimal_split"))
    kernels: list[str] = field(default_factory=lambda: get_config_value("visualization", "kernels", default=[]))

    @staticmethod
    def compute_weighted_area(entry):
        return (
            entry["utilization"]["util_bram"] * WBRAM
            + entry["utilization"]["util_dsp"] * WDSP
            + entry["utilization"]["util_ff"] * WFF
            + entry["utilization"]["util_lut"] * WLUT
        )

    def plot_pareto_samples(self):
        fig, axs = plt.subplots(3, 4, figsize=(20, 15))
        plt.subplots_adjust(wspace=0.3, hspace=0.4)
        axs = axs.flatten()

        for index, kernel in enumerate(self.kernels):
            ax = axs[index]
            full_pareto_path = os.path.join(self.pareto_full_dir, f"{kernel}_merged.json")
            try:
                with open(full_pareto_path, "r", encoding="utf-8") as file:
                    full_data = json.load(file)
            except FileNotFoundError:
                print(f"Background file not found: {full_pareto_path}")
                ax.set_title(f"{kernel} (No Data)", fontsize=12)
                continue

            latencies_full = []
            areas_full = []
            for entry in full_data:
                latency = entry["latency"]
                if latency == 1e8:
                    continue
                latencies_full.append(latency)
                areas_full.append(self.compute_weighted_area(entry))

            latencies_full = np.array(latencies_full)
            areas_full = np.array(areas_full)
            lat_max = np.max(latencies_full) if len(latencies_full) > 0 else 1.0
            area_max = np.max(areas_full) if len(areas_full) > 0 else 1.0

            ax.scatter(
                latencies_full / lat_max,
                areas_full / area_max,
                color="lightgray",
                s=20,
                alpha=0.3,
                zorder=1,
            )

            sample_path = os.path.join(self.pareto_samples_dir, kernel, f"{kernel}_diverse_split_33.json")
            try:
                with open(sample_path, "r", encoding="utf-8") as file:
                    sample_data = json.load(file)
            except FileNotFoundError:
                print(f"Sample file not found: {sample_path}")
                sample_data = []

            if sample_data:
                latencies_sample = np.array([entry["latency"] for entry in sample_data])
                areas_sample = np.array([self.compute_weighted_area(entry) for entry in sample_data])
                ax.scatter(
                    latencies_sample / lat_max,
                    areas_sample / area_max,
                    color="red",
                    marker="*",
                    s=150,
                    label="Diverse Sample (12 pts)",
                    zorder=4,
                    edgecolors="black",
                )

            ax.set_title(kernel, fontsize=14, weight="bold")
            if index >= 8:
                ax.set_xlabel("Normalized Latency", fontsize=12)
            if index % 4 == 0:
                ax.set_ylabel("Normalized Area", fontsize=12)
            ax.set_xlim(0, 1.05)
            ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle="--", alpha=0.5)
            if index == 0:
                ax.legend()

        fig.suptitle("Visualization of Diverse Sampling on Pareto Fronts", fontsize=20, weight="bold")
        plt.tight_layout()
        plt.show()


def plot_pareto_samples():
    return ParetoVisualizer().plot_pareto_samples()


if __name__ == "__main__":
    ParetoVisualizer().plot_pareto_samples()
