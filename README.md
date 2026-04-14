# CoT Pipeline

## Project Overview

This repository contains a generalized chain-of-thought (CoT) data generation pipeline for question-answer datasets.

The pipeline:
- Loads CSV or Parquet datasets.
- Maps dataset columns to canonical fields.
- Requires both `question` and `ground_truth` (or `answer`) as input.
- Optionally enriches metadata (`domain`, `task`, `language`) with an LLM mapper model.
- Generates model responses with rejection sampling.
- Verifies generated answers against ground truth.
- Saves output in Parquet format.

Supported domains:
- Chemistry
- Math
- Physics
- Biology
- Language

## Repository Structure

- `pipeline.py`: Main pipeline entry point.
- `config.yaml`: Model endpoints and pipeline settings.
- `requirements.txt`: Python dependencies.
- `Datasets/`: Input datasets (includes `demo_dataset.parquet`).
- `Generated_Dataset/`: Output directory for generated results.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/piyushkeshri8912/CoT-Pipeline.git
cd CoT-Pipeline
```

### 2. Create and activate a Python environment

Using venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Using conda (optional):

```bash
conda create -n cot-pipeline python=3.10 -y
conda activate cot-pipeline
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure model endpoints

Edit `config.yaml` and set valid values for:
- `mapper_model.url`, `mapper_model.model`, `mapper_model.api_key`
- `model.url`, `model.model`, `model.api_key`

Also review:
- `pipeline.output_dir` (default: `Generated_Dataset`)
- `pipeline.max_workers`
- `pipeline.max_attempts`
- `pipeline.timeout_s`

## Input Requirements

Input file must be `.csv` or `.parquet` with:
- `question` (required)
- `ground_truth` or `answer` (required)

Optional columns:
- `domain`
- `task`
- `source`
- `language`

If your column names differ, pass explicit mapping with `--column-map`.

## Run the Pipeline

### Demo run (recommended first)

```bash
python pipeline.py Datasets/demo_dataset.parquet --config config.yaml --dry-run
```

### Full run

```bash
python pipeline.py Datasets/demo_dataset.parquet --config config.yaml
```

### Generic run

```bash
python pipeline.py <input_file> --config config.yaml
```

### With explicit column mapping

```bash
python pipeline.py <input_file> \
  --config config.yaml \
  --column-map question=<question_col> answer=<answer_col>
```

### Useful options

```bash
# Process first 100 rows
python pipeline.py <input_file> --config config.yaml --head 100

# Use custom worker count
python pipeline.py <input_file> --config config.yaml --workers 20

# Restrict to selected domains
python pipeline.py <input_file> --config config.yaml --domains Math Physics Biology
```

## Output

Each run creates files under:

`Generated_Dataset/<dataset_key>_<model_name>/`

Artifacts:
- `cot_output.parquet`: Row-wise raw pipeline records (accepted + rejected).
- `<dataset_key>.parquet`: Final accepted standardized dataset.
- `rejection_log.json`: Rejected-row summary.

## Dependencies

Dependencies are managed through `requirements.txt`:
- pandas
- pyarrow
- requests
- PyYAML
- urllib3

## Troubleshooting

- Missing required columns: pass `--column-map`.
- API errors: verify URL/model/api key values in `config.yaml`.
- Parquet write issues: ensure `pyarrow` is installed.
- First-time checks: run with `--dry-run` before full generation.
