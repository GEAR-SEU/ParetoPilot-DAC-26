import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import psutil

from config_loader import get_config_value


@dataclass
class HLSRunner:
    timeout_seconds: int
    poll_interval_seconds: int

    @classmethod
    def from_config(cls) -> "HLSRunner":
        return cls(
            timeout_seconds=get_config_value("hls", "timeout_seconds", default=1200),
            poll_interval_seconds=3,
        )

    @staticmethod
    def build_report_path(xml_file_path_head: str, project_name: str, execute_file: str) -> str:
        return os.path.join(
            xml_file_path_head,
            project_name,
            "solution/syn/report",
            f"{execute_file}_csynth.xml",
        )

    def get_project_latency(self, xml_file_path_head, project_name, execute_file):
        xml_file_path = self.build_report_path(xml_file_path_head, project_name, execute_file)
        latency, utilization, total = self.parse_xml_file(xml_file_path)
        print("Synthesized Latency:", latency)
        print("Resource Utilization:", utilization)
        return latency, utilization, total

    @staticmethod
    def parse_xml_file(xml_file_path):
        path = Path(xml_file_path)
        if not path.exists():
            print(f"File {xml_file_path} does not exist")
            return None, None, None

        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            print(f"XML parse failed: {xml_file_path} is empty or invalid")
            return None, None, None
        except Exception as exc:
            print(f"Unexpected error: {exc}")
            return None, None, None

        latency = None
        bram_total = dsp_total = ff_total = lut_total = None
        bram_available = dsp_available = ff_available = lut_available = None

        for perf_est in root.findall("PerformanceEstimates"):
            for latency_elem in perf_est.findall("SummaryOfOverallLatency"):
                for child in latency_elem:
                    if "Average-caseLatency" in child.tag:
                        latency = int(child.text)

        for area_est in root.findall("AreaEstimates"):
            for resource in area_est.findall("Resources"):
                for child in resource:
                    if "BRAM_18K" in child.tag:
                        bram_total = int(child.text)
                    elif "DSP" in child.tag:
                        dsp_total = int(child.text)
                    elif "FF" in child.tag:
                        ff_total = int(child.text)
                    elif "LUT" in child.tag:
                        lut_total = int(child.text)

            for resource_usage in area_est.findall("AvailableResources"):
                for child in resource_usage:
                    if "BRAM_18K" in child.tag:
                        bram_available = int(child.text)
                    elif "DSP" in child.tag:
                        dsp_available = int(child.text)
                    elif "FF" in child.tag:
                        ff_available = int(child.text)
                    elif "LUT" in child.tag:
                        lut_available = int(child.text)

        required_values = [
            latency,
            bram_total,
            dsp_total,
            ff_total,
            lut_total,
            bram_available,
            dsp_available,
            ff_available,
            lut_available,
        ]
        if any(value is None for value in required_values):
            return None, None, None

        utilization = {
            "util_bram": bram_total / bram_available,
            "util_dsp": dsp_total / dsp_available,
            "util_ff": ff_total / ff_available,
            "util_lut": lut_total / lut_available,
        }
        totals = {
            "total_bram_18k": bram_total,
            "total_dsp": dsp_total,
            "total_ff": ff_total,
            "total_lut": lut_total,
        }
        return latency, utilization, totals

    @staticmethod
    def kill_vitis_process():
        for proc in psutil.process_iter():
            try:
                if proc.name() in {"vitis_hls.exe", "clang.exe"}:
                    proc.kill()
                    print("Killed a Vitis-related process.")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def run_hls(self, script_path, vitis_hls_run, cwd):
        vitis_file = os.path.join(cwd, "vitis_hls.log")
        start_time = time.time()

        with open(vitis_file, "w", encoding="utf-8") as log_file:
            print("HLS synthesis start")
            process = subprocess.Popen(
                [vitis_hls_run, script_path],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=cwd,
            )
            while True:
                elapsed_time = time.time() - start_time
                if process.poll() is not None:
                    break
                if elapsed_time > self.timeout_seconds:
                    print(f"Timeout: Vitis HLS for {script_path} is hanging. Terminating the process.")
                    log_file.write("Terminated due to timeout\n")
                    try:
                        self.kill_vitis_process()
                        process.terminate()
                    except Exception as exc:
                        print(f"Error killing process tree: {exc}")
                    break
                time.sleep(self.poll_interval_seconds)

            process.wait()
            print(f"Vitis HLS process for {script_path} completed.")

    @staticmethod
    def delete_project_folder(cwd, project_name):
        project_folder = os.path.join(cwd, project_name)
        vitis_file = os.path.join(cwd, "vitis_hls.log")

        if os.path.exists(project_folder):
            shutil.rmtree(project_folder)
            print(f"Successfully deleted the project folder: {project_folder}\n")
        else:
            print(f"Project folder {project_folder} does not exist.")

        if os.path.exists(vitis_file):
            os.remove(vitis_file)


DEFAULT_HLS_RUNNER = HLSRunner.from_config()


def get_project_latency(xml_file_path_head, project_name, execute_file):
    return DEFAULT_HLS_RUNNER.get_project_latency(xml_file_path_head, project_name, execute_file)


def parse_xml_file(xml_file_path):
    return DEFAULT_HLS_RUNNER.parse_xml_file(xml_file_path)


def kill_vitis_process():
    return DEFAULT_HLS_RUNNER.kill_vitis_process()


def run_hls(script_path, vitis_hls_run, cwd):
    return DEFAULT_HLS_RUNNER.run_hls(script_path, vitis_hls_run, cwd)


def delete_project_folder(cwd, project_name):
    return DEFAULT_HLS_RUNNER.delete_project_folder(cwd, project_name)
