import argparse
import json
import os
from dataclasses import dataclass, field

import tool.feature_extractor as feature_extractor
import tool.hls_tools as hls_tools
import tool.sampling as sampling
from config_loader import get_config_value


@dataclass
class DSEWorkflowConfig:
    project_root: str
    config_root: str
    vitis_hls_run: str
    output_dir: str
    llm_backend: str = "hf"
    hf_model: str | None = None
    projects: list[str] = field(default_factory=list)
    init_samples: int = 3

    @classmethod
    def from_config(cls) -> "DSEWorkflowConfig":
        return cls(
            project_root=get_config_value("dse", "project_root", default="benchmark"),
            config_root=get_config_value("dse", "config_root", default="design_space_config"),
            vitis_hls_run=get_config_value("dse", "vitis_hls_bin", default="vitis_hls"),
            output_dir=get_config_value("dse", "output_dir", default="dse_results"),
            llm_backend=get_config_value("llm", "backend", default="hf"),
            hf_model=get_config_value("llm", "hf_model", default="XXXiong/ParetoPilot"),
            projects=get_config_value("dse", "projects", default=[]),
            init_samples=get_config_value("dse", "init_samples", default=3),
        )


@dataclass
class DSEWorkflow:
    config: DSEWorkflowConfig

    def _persist_results(self, execute_file: str, results) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        output_path = os.path.join(self.config.output_dir, f"dse_results_{execute_file}.json")
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(results, json_file, indent=4)

    def perform_dse(self, execute_project, execute_file, config_loop_array_path):
        project_base_name = os.path.basename(execute_project)
        project_head_path = os.path.join(execute_project, f"{execute_file}.c")

        llm_result = sampling.llm_sampling_with_metadata(
            project_head_path,
            config_loop_array_path,
            self.config.init_samples,
            backend=self.config.llm_backend,
            hf_model=self.config.hf_model,
        )
        config_set = llm_result["combinations"]
        print(f"Total LLM sampled combinations: {len(config_set)}")

        results = []
        for iteration, combination in enumerate(config_set[: self.config.init_samples]):
            loop_combination, array_combination = combination
            print(f"loop_combination: {loop_combination}")
            print(f"array_combination: {array_combination}")

            trial_project_name = f"{project_base_name}_trial_{iteration}"
            script_path = execute_project + '/' + "dataset_hls.tcl"
            script_path = os.path.abspath(script_path).replace('\\', '/')
            feature_extractor.modify_tcl_file(
                script_path,
                loop_combination,
                array_combination,
                trial_project_name,
                project_base_name,
            )

            hls_tools.run_hls(script_path, self.config.vitis_hls_run, execute_project)
            latency, utilization, total = hls_tools.get_project_latency(
                execute_project,
                trial_project_name,
                execute_file,
            )
            hls_tools.delete_project_folder(execute_project, trial_project_name)

            if latency is None or utilization is None:
                latency, utilization, total = 1e8, 1e8, None

            result_entry = {
                "index": iteration,
                "loop_combination": loop_combination,
                "array_combination": array_combination,
                "latency": latency,
                "utilization": utilization,
                "total": total,
            }
            results.append(result_entry)
            self._persist_results(execute_file, results)

        return results

    def run(self):
        for project in self.config.projects:
            source_project_dir = self.config.project_root + '/' + project
            config_loop_array = os.path.join(self.config.config_root, f"design_space_config_{project}.json")
            self.perform_dse(source_project_dir, project, config_loop_array)


def perform_dse(execute_project, execute_file, vitis_hls_run, config_loop_array_path):
    config = DSEWorkflowConfig.from_config()
    config.vitis_hls_run = vitis_hls_run
    workflow = DSEWorkflow(config)
    return workflow.perform_dse(execute_project, execute_file, config_loop_array_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ParetoPilot design space exploration workflow.")
    parser.add_argument(
        "--llm-backend",
        choices=("hf", "api"),
        default=None,
        help="Choose the LLM backend. Defaults to the config value, or 'hf' when omitted.",
    )
    parser.add_argument(
        "--hf-model",
        default=None,
        help="Hugging Face model name or repo id, for example XXXiong/ParetoPilot.",
    )
    return parser


def build_runtime_config(args: argparse.Namespace) -> DSEWorkflowConfig:
    config = DSEWorkflowConfig.from_config()
    if args.llm_backend is not None:
        config.llm_backend = args.llm_backend
    if args.hf_model is not None:
        config.hf_model = args.hf_model
    return config


if __name__ == "__main__":
    arguments = build_arg_parser().parse_args()
    DSEWorkflow(build_runtime_config(arguments)).run()
