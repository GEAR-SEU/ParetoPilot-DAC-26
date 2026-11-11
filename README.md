# ParetoPilot: Global Optimization Reasoning on HLS Design Space Exploration with LLMs
> 🌟 **Accepted by DAC 2026**

[![Paper](https://img.shields.io/badge/Paper-DAC%202026-blue.svg)](https://dl.acm.org/doi/abs/10.1145/3770743.3804051)
[![Model](https://img.shields.io/badge/🤗_Hugging_Face-ParetoPilot-ffc107.svg)](https://huggingface.co/XXXiong/ParetoPilot)

ParetoPilot is a lightweight, automated workflow for LLM-guided high-level synthesis (HLS) design space exploration (DSE). By instilling LLMs with expert-like global optimization reasoning, ParetoPilot efficiently navigates the massive combinatorial space of HLS directives to rapidly shape broad and concave Pareto fronts.

![ParetoPilot Workflow](img/ParetoPilot_workflow.png)

## 📂 Structure

- `src/DSE_LLM.py`: main entry for ParetoPilot DSE runs.
- `src/tool/`: utilities for sampling, HLS execution, feature extraction, and evaluation.
- `src/design_space_config/`: benchmark-specific pragma search-space definitions.
- `src/dataset_construction/`: dataset construction scripts for directive reasoning, Pareto reasoning, and QoR reasoning tasks.
- `src/dissimilarity_sampling/`: scripts for Pareto front filtering, diverse subset generation, and sample visualization.
- `dse_results/`: generated DSE outputs written by the workflow at runtime.

## ⚙️ Configuration

Update `config.yaml` before running. The main related keys are:

- `llm.backend`: default backend, `hf` or `api`
- `llm.hf_model`: default `XXXiong/ParetoPilot` or local model path
- `llm.api_key`, `llm.api_base`, `llm.model`: API-mode settings
- `dse.vitis_hls_bin`: path to your local `vitis_hls` executable or batch file, which must be updated for your machine
- `dse.projects`: list of benchmark projects to run. Supported values are `aes`, `bbgemm`, `dft`, `fir`, `gemm`, `kernel_2mm`, `kernel_3mm`, `kernel_symm`, `kernel_syr2k`, `kernel_syrk`, `matmul`, and `sort`

## 🏃 Quickstart

Before installing dependencies or running the workflow, check the configuration in `config.yaml` and review the `Configuration` section above.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main workflow from the repository root. By default, ParetoPilot downloads model from  Hugging Face and uses it for inference:

```bash
python src/DSE_LLM.py
```

Override the default Hugging Face model from the command line if needed:

```bash
python src/DSE_LLM.py --hf-model XXXiong/ParetoPilot
```

Switch back to the API-based workflow with a command-line flag:

```bash
python src/DSE_LLM.py --llm-backend api
```

## 📖 License

This repository is released under the Apache License 2.0. See [LICENSE](LICENSE).
