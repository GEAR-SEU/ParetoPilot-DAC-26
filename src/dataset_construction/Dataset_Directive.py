import os

from common import BaseDatasetProcessor
from config_loader import get_config_value


class DirectiveDatasetProcessor(BaseDatasetProcessor):
    log_filename = "directive.log"
    max_workers = 10
    input_extension = ".tcl"
    project_names = get_config_value("dataset_construction", "directive", "projects", default=[])

    def build_prompt(self, unoptimized_hls_design, unoptimized_design_report, tcl_optimization_directives):
        return f"""You are an expert HLS engineer. Your task is to analyze a given HLS design and justify the application of a specific set of optimization directives as a comprehensive optimization strategy.
I will provide you with:

- The HLS design code.
- The performance report for the original unoptimized design.
- The HLS optimization directives as Tcl commands.
- The optimized design performance report for reference.

Your task is to provide a structured qualitative analysis that explains why this directive set is a sound optimization strategy for the given code.

Original unoptimized HLS design:
```cpp
{unoptimized_hls_design}
```

Original design performance report:
```text
{unoptimized_design_report}
```

Optimization directives and optimized design report:
```tcl
{tcl_optimization_directives}
```
"""

    def get_system_prompt(self):
        return (
            "You are an expert in HLS design optimization and design space exploration. "
            "Provide clear reasoning for the supplied optimization directives."
        )

    def get_task_directory(self, project):
        return os.path.join("repo_tcl_deepseek", project)

    def read_task_inputs(self, project, input_filename):
        design_path = os.path.join("trainset", project, f"{project}.c")
        report_path = os.path.join("trainset", project, f"results_{project}.json")
        directive_path = os.path.join("repo_tcl", project, input_filename)

        with open(design_path, "r", encoding="utf-8") as file:
            unoptimized_hls_design = file.read()
        with open(report_path, "r", encoding="utf-8") as file:
            unoptimized_design_report = file.read()
        with open(directive_path, "r", encoding="utf-8") as file:
            tcl_optimization_directives = file.read()

        return unoptimized_hls_design, unoptimized_design_report, tcl_optimization_directives


if __name__ == "__main__":
    DirectiveDatasetProcessor().run()
