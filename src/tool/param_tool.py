from dataclasses import dataclass

import numpy as np


@dataclass
class ParetoMetrics:
    @staticmethod
    def adrs(reference_set, approximate_set):
        min_dist_sum = 0
        for reference_point in reference_set:
            distances = [
                ParetoMetrics.p2p_distance_2d(reference_point, approximate_point)
                for approximate_point in approximate_set
            ]
            min_dist_sum += min(distances)

        return min_dist_sum / len(reference_set)

    @staticmethod
    def p2p_distance_2d(ref_pt, app_pt):
        x = (float(app_pt["latency"]) - float(ref_pt["latency"])) / float(ref_pt["latency"])
        y = (float(app_pt["area"]) - float(ref_pt["area"])) / float(ref_pt["area"])
        return max([0, x, y])

    @staticmethod
    def is_pareto_optimal(latencies, areas, idx):
        latency_i = latencies[idx]
        area_i = areas[idx]
        for candidate_index in range(len(latencies)):
            if candidate_index != idx and latencies[candidate_index] < latency_i and areas[candidate_index] < area_i:
                return False
        return True

    @staticmethod
    def extract_features(entries):
        loop_names = set()
        array_dimensions = set()

        for entry in entries:
            for loop in entry["loop_combination"]:
                loop_names.add(loop["name"])
            for array_entry in entry.get("array_combination") or []:
                array_dimensions.add((array_entry["name"], array_entry["dim"]))

        sorted_loops = sorted(loop_names)
        sorted_arrays = sorted(array_dimensions, key=lambda item: (item[0], item[1]))

        features = []
        for entry in entries:
            feature_vector = []

            loop_dict = {loop["name"]: loop for loop in entry["loop_combination"]}
            for loop_name in sorted_loops:
                loop = loop_dict.get(loop_name, {})
                feature_vector.append(loop.get("pipeline", 0))
                feature_vector.append(loop.get("unroll", 0))

            array_dict = {
                (array_entry["name"], array_entry["dim"]): array_entry
                for array_entry in (entry.get("array_combination") or [])
            }
            for name, dim in sorted_arrays:
                array_entry = array_dict.get((name, dim), {})
                feature_vector.append(array_entry.get("type", 0))
                feature_vector.append(array_entry.get("factor", 0))

            features.append(feature_vector)

        return features

    @staticmethod
    def extract_features_lattice(entries):
        return [entry["combination"] for entry in entries]


DEFAULT_PARETO_METRICS = ParetoMetrics()


def adrs(reference_set, approximate_set):
    return DEFAULT_PARETO_METRICS.adrs(reference_set, approximate_set)


def p2p_distance_2d(ref_pt, app_pt):
    return DEFAULT_PARETO_METRICS.p2p_distance_2d(ref_pt, app_pt)


def is_pareto_optimal(latencies, areas, idx):
    return DEFAULT_PARETO_METRICS.is_pareto_optimal(latencies, areas, idx)


def extract_features(entries):
    return DEFAULT_PARETO_METRICS.extract_features(entries)


def extract_features_lattice(entries):
    return DEFAULT_PARETO_METRICS.extract_features_lattice(entries)
