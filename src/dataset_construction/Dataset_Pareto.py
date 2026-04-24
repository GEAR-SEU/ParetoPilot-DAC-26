import os

from common import BaseDatasetProcessor
from config_loader import get_config_value


class ParetoDatasetProcessor(BaseDatasetProcessor):
    log_filename = "pareto.log"
    max_workers = 10
    input_extension = ".md"
    project_names = get_config_value("dataset_construction", "pareto", "projects", default=[])

    def build_prompt(self, kernel_code, pragma_config, output_pragma):
        return f"""You are an expert High-Level Synthesis optimization engineer with deep knowledge of design space exploration.
Your task is to analyze the given optimization directives and describe how they form a Pareto front.

HLS kernel code:
```cpp
{kernel_code}
```

Available pragma parameters:
```json
{pragma_config}
```

Proposed HLS configuration:
```python
{output_pragma}
```
"""

    def get_system_prompt(self):
        return (
            "You are an expert High-Level Synthesis optimization engineer. "
            "Provide an expert justification for the supplied pragma configuration."
        )

    def get_task_directory(self, project):
        return os.path.join("DSE_Data/optimal_split_tuple_10000", project)

    def read_task_inputs(self, project, input_filename):
        design_path = os.path.join("trainset", project, f"{project}.c")
        config_path = f"trainset/loops_arrays_config_pruned/loop_array_pragma_config_{project}_pruned.json"
        directive_path = os.path.join("DSE_Data/optimal_split_tuple_10000", project, input_filename)

        with open(design_path, "r", encoding="utf-8") as file:
            kernel_code = file.read()
        with open(config_path, "r", encoding="utf-8") as file:
            pragma_config = file.read()
        with open(directive_path, "r", encoding="utf-8") as file:
            output_pragma = file.read()

        return kernel_code, pragma_config, output_pragma


if __name__ == "__main__":
    ParetoDatasetProcessor().run()
