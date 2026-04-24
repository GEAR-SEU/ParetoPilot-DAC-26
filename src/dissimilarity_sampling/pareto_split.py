import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config_loader import get_config_value


WLUT = 0.3
WFF = 0.25
WDSP = 0.3
WBRAM = 0.05


@dataclass
class DiverseSampler:
    input_dir: str = field(default_factory=lambda: get_config_value("pareto_split", "input_dir", default="DSE_Data/optimal_500"))
    output_dir: str = field(default_factory=lambda: get_config_value("pareto_split", "output_dir", default="DSE_Data/optimal_split_10000"))
    kernels: list[str] = field(default_factory=lambda: get_config_value("pareto_split", "kernels", default=[]))
    num_output_files_per_kernel: int = 1000
    num_points_to_select: int = 12

    @staticmethod
    def compute_weighted_area(entry):
        utilization = entry.get("utilization", {})
        return (
            utilization.get("util_bram", 0) * WBRAM
            + utilization.get("util_dsp", 0) * WDSP
            + utilization.get("util_ff", 0) * WFF
            + utilization.get("util_lut", 0) * WLUT
        )

    @staticmethod
    def calculate_crowding_distance(points):
        num_points = len(points)
        if num_points <= 2:
            for point in points:
                point["crowding_distance"] = float("inf")
            return points

        for point in points:
            point["crowding_distance"] = 0.0

        for key in ["norm_latency", "norm_area"]:
            points.sort(key=lambda point: point[key])
            points[0]["crowding_distance"] = float("inf")
            points[-1]["crowding_distance"] = float("inf")

            min_val = points[0][key]
            max_val = points[-1][key]
            value_range = max_val - min_val
            if value_range == 0:
                continue

            for index in range(1, num_points - 1):
                distance = points[index + 1][key] - points[index - 1][key]
                points[index]["crowding_distance"] += distance / value_range

        return points

    def select_diverse_points(self, designs, num_to_select):
        processed_designs = []
        for index, entry in enumerate(designs):
            processed_designs.append(
                {
                    "id": index,
                    "latency": entry["latency"],
                    "area": self.compute_weighted_area(entry),
                    "original_entry": entry,
                }
            )

        latencies = [point["latency"] for point in processed_designs]
        areas = [point["area"] for point in processed_designs]
        min_lat, max_lat = min(latencies), max(latencies)
        min_area, max_area = min(areas), max(areas)

        for point in processed_designs:
            point["norm_latency"] = (point["latency"] - min_lat) / (max_lat - min_lat) if max_lat > min_lat else 0
            point["norm_area"] = (point["area"] - min_area) / (max_area - min_area) if max_area > min_area else 0

        selected_points_map = {}
        min_lat_point = min(processed_designs, key=lambda point: (point["norm_latency"], point["norm_area"]))
        min_area_point = min(processed_designs, key=lambda point: (point["norm_area"], point["norm_latency"]))
        knee_point = min(
            processed_designs,
            key=lambda point: np.sqrt(point["norm_latency"] ** 2 + point["norm_area"] ** 2),
        )
        selected_points_map[min_lat_point["id"]] = min_lat_point
        selected_points_map[min_area_point["id"]] = min_area_point
        selected_points_map[knee_point["id"]] = knee_point

        processed_designs = self.calculate_crowding_distance(processed_designs)
        num_to_sample = num_to_select - len(selected_points_map)

        if num_to_sample > 0:
            candidates = [point for point in processed_designs if point["id"] not in selected_points_map]
            if candidates:
                weights = [point["crowding_distance"] for point in candidates]
                max_finite_weight = max([weight for weight in weights if weight != float("inf")], default=1.0)
                weights = [weight if weight != float("inf") else max_finite_weight * 10 for weight in weights]

                if len(candidates) <= num_to_sample:
                    sampled_points = candidates
                else:
                    sampled_points = random.choices(candidates, weights=weights, k=num_to_sample)

                for point in sampled_points:
                    selected_points_map[point["id"]] = point

        return [point["original_entry"] for point in selected_points_map.values()]

    def create_diverse_splits_for_kernel(self, kernel):
        print(f"Creating diverse samples for kernel: {kernel}")
        input_path = os.path.join(self.input_dir, f"pareto_optimal_{kernel}.json")
        try:
            with open(input_path, "r", encoding="utf-8") as file:
                all_designs = json.load(file)
        except FileNotFoundError:
            print(f"Input file not found: {input_path}")
            return

        if len(all_designs) < self.num_points_to_select:
            print(f"Kernel {kernel} has only {len(all_designs)} design points, fewer than {self.num_points_to_select}.")
            return

        kernel_output_dir = os.path.join(self.output_dir, kernel)
        os.makedirs(kernel_output_dir, exist_ok=True)

        print(f"Generating {self.num_output_files_per_kernel} sample files for {kernel}...")
        for index in range(self.num_output_files_per_kernel):
            selected_designs = self.select_diverse_points(all_designs, self.num_points_to_select)
            output_filename = f"{kernel}_diverse_split_{index}.json"
            output_path = os.path.join(kernel_output_dir, output_filename)
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(selected_designs, file, indent=2)

        print(f"Generated {self.num_output_files_per_kernel} files in {kernel_output_dir}")

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        print("Starting diverse sample generation...")
        for kernel in self.kernels:
            self.create_diverse_splits_for_kernel(kernel)
            print("-" * 50)
        print("All kernels processed.")
        print(f"Output directory: {os.path.abspath(self.output_dir)}")


def calculate_crowding_distance(points):
    return DiverseSampler.calculate_crowding_distance(points)


def select_diverse_points(designs, num_to_select):
    return DiverseSampler().select_diverse_points(designs, num_to_select)


def create_diverse_splits_for_kernel(kernel):
    return DiverseSampler().create_diverse_splits_for_kernel(kernel)


if __name__ == "__main__":
    DiverseSampler().run()
