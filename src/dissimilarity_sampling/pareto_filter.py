import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config_loader import get_config_value


WLUT = 0.3
WFF = 0.25
WDSP = 0.3
WBRAM = 0.05


@dataclass
class ParetoFilter:
    input_dir: str = field(default_factory=lambda: get_config_value("pareto_filter", "input_dir", default="DSE_Data"))
    output_dir: str = field(default_factory=lambda: get_config_value("pareto_filter", "output_dir", default="DSE_Data/optimal_50"))
    kernels: list[str] = field(default_factory=lambda: get_config_value("pareto_filter", "kernels", default=[]))
    num_optimal_points_to_find: int = 100

    @staticmethod
    def compute_weighted_area(entry):
        utilization = entry.get("utilization", {})
        return (
            utilization.get("util_bram", 0) * WBRAM
            + utilization.get("util_dsp", 0) * WDSP
            + utilization.get("util_ff", 0) * WFF
            + utilization.get("util_lut", 0) * WLUT
        )

    def extract_and_save_pareto_points(self, kernel, num_points_to_find):
        print(f"Processing kernel: {kernel}")
        input_path = os.path.join(self.input_dir, f"{kernel}_merged.json")
        try:
            with open(input_path, "r", encoding="utf-8") as file:
                all_data = json.load(file)
        except FileNotFoundError:
            print(f"Input file not found: {input_path}")
            return

        processed_points = []
        for index, entry in enumerate(all_data):
            if entry.get("latency") == 1e8:
                continue

            processed_points.append(
                {
                    "id": index,
                    "latency": entry["latency"],
                    "area": self.compute_weighted_area(entry),
                    "original_entry": entry,
                }
            )

        if not processed_points:
            print(f"No valid design points found for kernel: {kernel}")
            return

        final_results = []
        candidate_points = processed_points
        while len(final_results) < num_points_to_find and candidate_points:
            current_front = []
            for point_a in candidate_points:
                is_dominated = False
                for point_b in candidate_points:
                    if point_a["id"] == point_b["id"]:
                        continue
                    if (
                        (point_b["latency"] <= point_a["latency"] and point_b["area"] < point_a["area"])
                        or (point_b["latency"] < point_a["latency"] and point_b["area"] <= point_a["area"])
                    ):
                        is_dominated = True
                        break

                if not is_dominated:
                    current_front.append(point_a)

            if not current_front:
                print("No additional Pareto front points found.")
                break

            current_front.sort(key=lambda point: point["latency"])
            for point in current_front:
                if len(final_results) >= num_points_to_find:
                    break
                final_results.append(point["original_entry"])

            front_ids = {point["id"] for point in current_front}
            candidate_points = [point for point in candidate_points if point["id"] not in front_ids]
            print(f"Collected {len(final_results)} Pareto points so far.")

        if not final_results:
            print(f"No Pareto-optimal points found for kernel: {kernel}")
            return

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"pareto_optimal_{kernel}.json")
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(final_results, file, indent=2)
        print(f"Saved {len(final_results)} Pareto points to {output_path}")

    def run(self):
        for kernel in self.kernels:
            self.extract_and_save_pareto_points(kernel, self.num_optimal_points_to_find)
            print("-" * 50)
        print("All kernels processed.")


def extract_and_save_pareto_points(kernel, num_points_to_find):
    return ParetoFilter().extract_and_save_pareto_points(kernel, num_points_to_find)


if __name__ == "__main__":
    ParetoFilter().run()
