
import os
from dataclasses import dataclass
from config_loader import get_config_value


@dataclass
class DirectiveBuilder:
    default_part: str
    default_clock_period: int

    @classmethod
    def from_config(cls) -> "DirectiveBuilder":
        return cls(
            default_part=get_config_value("tcl", "default_part", default="xczu7ev-ffvc1156-2-e"),
            default_clock_period=get_config_value("tcl", "default_clock_period", default=10),
        )

    def modify_tcl_file(self, script_path, loop_combination, array_combination, project_name, function_name):
        tcl_content = []
        parent_dir = os.path.dirname(script_path)

        tcl_content.append(f"open_project {project_name}\n\n")
        for file_name in os.listdir(parent_dir):
            if file_name.endswith((".c", ".cpp", ".h")):
                tcl_content.append(f"add_files {file_name}\n")

        tcl_content.append(f"set_top {function_name}\n")
        tcl_content.append("\nopen_solution solution\n")
        tcl_content.append(f"set_part {{{self.default_part}}}\n")
        tcl_content.append(f"create_clock -period {self.default_clock_period} -name default\n\n")

        if array_combination:
            for directives in array_combination:
                array_name = directives["name"]
                type_value = directives["type"]
                dim_value = directives["dim"]
                factor_value = directives["factor"]

                if type_value == 0:
                    type_pragma = "complete"
                elif type_value == 1:
                    type_pragma = "block"
                else:
                    type_pragma = "cyclic"

                if type_value == 0:
                    tcl_content.append(
                        f"set_directive_array_partition -type {type_pragma} -dim {dim_value} "
                        f"\"{project_name}\" {array_name}\n"
                    )
                elif factor_value not in (None, 0):
                    tcl_content.append(
                        f"set_directive_array_partition -type {type_pragma} -factor {factor_value} "
                        f"-dim {dim_value} \"{project_name}\" {array_name}\n"
                    )
        else:
            print("No array combinations to process.")

        tcl_content.append("\n")

        if loop_combination:
            for directives in loop_combination:
                loop_name = directives["name"]

                if "pipeline" in directives:
                    pipeline_value = directives["pipeline"]
                    if pipeline_value == 1:
                        tcl_content.append(f"set_directive_pipeline \"{project_name}/{loop_name}\"\n")
                    else:
                        tcl_content.append(f"set_directive_pipeline -off \"{project_name}/{loop_name}\"\n")

                if "unroll" in directives:
                    unroll_value = directives["unroll"]
                    if unroll_value != 0:
                        tcl_content.append(f"set_directive_unroll -factor {unroll_value} \"{project_name}/{loop_name}\"\n")
                    else:
                        tcl_content.append("\n")

        tcl_content.append("\ncsynth_design\n")
        tcl_content.append("exit\n")

        with open(script_path, "w", encoding="utf-8") as file:
            file.writelines(tcl_content)


DEFAULT_DIRECTIVE_BUILDER = DirectiveBuilder.from_config()

def modify_tcl_file(script_path, loop_combination, array_combination, project_name, function_name):
    return DEFAULT_DIRECTIVE_BUILDER.modify_tcl_file(script_path, loop_combination, array_combination, project_name, function_name)