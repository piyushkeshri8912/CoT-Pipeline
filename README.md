# Team 3 CoT Data Pipeline

## Project Overview

This project contains a generalized chain-of-thought (CoT) generation pipeline for question-answer datasets.

The pipeline:
- Loads input data from CSV or Parquet.
- Maps dataset columns to canonical fields.
- Requires both `question` and `ground_truth` (or `answer`) columns.
- Optionally enriches metadata (`domain`, `task`, `language`) using an LLM mapper model.
- Generates answers with rejection sampling.
- Validates generated answers against ground truth.
- Saves outputs in Parquet format.

Supported domains include:
- Chemistry
- Math
- Physics
- Biology
- Language

## Folder Structure (Team_3)

- `pipeline.py`: Main pipeline script.
- `config.yaml`: Model and pipeline configuration.
- `requirements.txt`: Python dependencies.
- `Generated_Dataset/`: Default output location for run artifacts.

## Dependencies

Dependencies are listed in `requirements.txt`:
- pandas
- pyarrow
- requests
- PyYAML
- urllib3

## Installation

### 1. Move to workspace root

```bash
cd /projects/data/datasets/code_post_training_data
```

### 2. (Recommended) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

If you use conda, you can activate your environment instead:

```bash
conda activate base
```

### 3. Install dependencies

```bash
pip install -r Team_3/requirements.txt
```

## Configuration

Edit `Team_3/config.yaml` to set:
- `mapper_model`: used for schema mapping and metadata enrichment.
- `model`: used for CoT generation.
- `pipeline.output_dir`: default is `Team_3/Generated_Dataset`.
- `pipeline.max_workers`, `max_attempts`, `timeout_s`, etc.

## Input Requirements

The input file must be CSV or Parquet and contain:
- `question` (required)
- `ground_truth` or `answer` (required)

Optional columns:
- `domain`
- `task`
- `source`
- `language`

If your column names are different, pass explicit mapping with `--column-map`.

## Running the Pipeline

### Basic run

```bash
python Team_3/pipeline.py <input_file> --config Team_3/config.yaml
```

Demo dataset example:

```bash
python Team_3/pipeline.py Team_3/Datasets/demo_dataset.parquet --config Team_3/config.yaml
```

Dry-run with demo dataset:

```bash
python Team_3/pipeline.py Team_3/Datasets/demo_dataset.parquet --config Team_3/config.yaml --dry-run
```

### Run with explicit column mapping

```bash
python Team_3/pipeline.py <input_file> \
  --config Team_3/config.yaml \
  --column-map question=<question_col> answer=<answer_col>
```

Note: `answer` is automatically mapped to `ground_truth`.

### Dry run (no model calls)

```bash
python Team_3/pipeline.py <input_file> --config Team_3/config.yaml --dry-run
```

### Limit rows for quick testing

```bash
python Team_3/pipeline.py <input_file> --config Team_3/config.yaml --head 100
```

### Control parallelism

```bash
python Team_3/pipeline.py <input_file> --config Team_3/config.yaml --workers 20
```

### Filter by domain

```bash
python Team_3/pipeline.py <input_file> \
  --config Team_3/config.yaml \
  --domains Math Physics Biology
```

## Output

For each run, outputs are written under:

`Team_3/Generated_Dataset/<dataset_key>_<model_name>/`

Generated files:
- `cot_output.parquet`: raw per-row pipeline results (accepted and rejected).
- `<dataset_key>.parquet`: final accepted standardized dataset.
- `rejection_log.json`: summary of rejected rows.


## Troubleshooting

- If required columns are missing, provide explicit `--column-map`.
- If API calls fail, verify model URL/API key in `config.yaml`.
- If Parquet writing fails, ensure `pyarrow` is installed.
- Use `--dry-run` to validate mapping and preprocessing before full execution.
