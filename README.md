# CoT + Segmentation Pipeline

## Overview

This repository contains two connected pipelines:

1. `segmentation_pipeline.py` (Step 2 from `task.md`)
- Reads raw OCR text from `OCR_output/`
- Segments text into `{question, answer}` pairs
- Writes per-file segmented outputs and a merged segmented dataset

2. `pipeline.py` (teacher generation + verification)
- Takes question + ground truth datasets
- Generates verified reasoning traces and final outputs

## Repository Structure

- `segmentation_pipeline.py`: Main Step-2 segmentation pipeline
- `pipeline.py`: CoT generation/verification pipeline
- `config.yaml`: Model endpoint configuration
- `requirements.txt`: Python dependencies
- `OCR_output/`: Raw OCR extracted text/parquet inputs
- `segmented_output/`: Step-2 segmentation outputs
- `Datasets/`: Training/eval datasets for `pipeline.py`
- `Generated_Dataset/`: Final CoT outputs

## Quick Start (Fresh Clone)

### 1. Clone repository

```bash
git clone https://github.com/piyushkeshri8912/CoT-Pipeline.git
cd CoT-Pipeline
```

### 2. Create Python environment

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

Edit `config.yaml` and set your model values:
- `model.url`
- `model.model`
- `model.api_key`

If you use `pipeline.py`, also set:
- `mapper_model.url`
- `mapper_model.model`
- `mapper_model.api_key`

## Step 2: Segmentation Pipeline (From OCR Output)

### Demo input included in repo

Demo OCR sample:
- `OCR_output/demo_ocr_output.parquet`

Run segmentation on demo input:

```bash
python segmentation_pipeline.py \
  --step segment \
  --input OCR_output/demo_ocr_output.parquet \
  --output-dir segmented_output/demo_ocr_output_segemented
```

Run segmentation on all OCR files:

```bash
python segmentation_pipeline.py --step segment --input-dir OCR_output --workers 8
```

### Step-2 output files

For each run, outputs include:
- per-file segmented parquet/csv
- merged segmented dataset: `segmented_qa.parquet`

## CoT Pipeline (`pipeline.py`)

### Demo run

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

### Explicit column mapping

```bash
python pipeline.py <input_file> \
  --config config.yaml \
  --column-map question=<question_col> answer=<answer_col>
```

## Dependencies

Dependencies in `requirements.txt` include:
- pandas
- pyarrow
- requests
- PyYAML
- urllib3
- openai
- PyMuPDF

## Troubleshooting

- Missing required input columns in `pipeline.py`: use `--column-map`.
- API failures: verify URL/model/api key values in `config.yaml`.
- PDF segmentation failures: ensure `PyMuPDF` is installed.
- First test: run segmentation on demo input before full dataset runs.
