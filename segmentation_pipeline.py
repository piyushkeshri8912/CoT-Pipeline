#!/usr/bin/env python3
"""
OCR Segmentation Pipeline
=========================

Primary goal:
- Read OCR dumps (parquet/PDF) from OCR_output/
- Segment each input into question-answer pairs
- Save one segmented file per source + one merged segmented dataset

Task scope:
- Segment extracted OCR text into {question, answer} pairs
- Preserve answer keys where available in OCR text

Quick examples:
    python segmentation_pipeline.py --step segment
    python segmentation_pipeline.py --step segment --input OCR_output/JA_maths_ocr.parquet
    python segmentation_pipeline.py --step segment --workers 8
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd  # type: ignore
import yaml

try:
    import fitz  # type: ignore
except ImportError:
    fitz = None

from openai import OpenAI  # type: ignore


# ============================================================================
#  1.  PATHS AND CONFIG
# ============================================================================

SCRIPT_DIR: Path = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH: Path = SCRIPT_DIR / "config.yaml"
DEFAULT_INPUT_DIR: Path = SCRIPT_DIR / "OCR_output"
DEFAULT_OUTPUT_DIR: Path = SCRIPT_DIR / "segmented_output"
DEFAULT_MERGED_DATASET_NAME = "segmented_qa.parquet"


def read_model_config(path: str, model_name: Optional[str] = None) -> Dict[str, str]:
    """Load model config from .yaml/.yml, .json, or legacy .txt format.

    Supported formats:
    - config.yaml with keys like model.url/model.model/model.api_key
    - models.json list of model entries (optional --model-name to select)
    - legacy text file containing model:, url:, api:
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        model_section = raw.get("model", {})
        if not isinstance(model_section, dict):
            raise ValueError("Invalid YAML format: 'model' section must be an object")

        model = str(model_section.get("model", "")).strip()
        url = str(model_section.get("url", "")).strip()
        api = str(model_section.get("api_key", "")).strip()

        if not model or not url:
            raise ValueError("YAML config must include model.model and model.url")

        return {"model": model, "url": url, "api": api}

    if suffix == ".json":
        with open(config_path, "r", encoding="utf-8") as fh:
            raw_json = json.load(fh)

        chosen: Dict[str, Any] = {}
        if isinstance(raw_json, list):
            if not raw_json:
                raise ValueError("JSON model list is empty")
            if model_name:
                for item in raw_json:
                    if str(item.get("name", "")).strip() == model_name:
                        chosen = item
                        break
                if not chosen:
                    raise ValueError(f"Model name '{model_name}' not found in {config_path}")
            else:
                chosen = raw_json[0]
        elif isinstance(raw_json, dict):
            chosen = raw_json
        else:
            raise ValueError("Unsupported JSON config structure")

        model = str(chosen.get("model", "")).strip()
        url = str(chosen.get("url", "")).strip()
        api = str(chosen.get("api_key", chosen.get("api", ""))).strip()

        if not model or not url:
            raise ValueError("JSON config must include model and url")

        return {"model": model, "url": url, "api": api}

    # Legacy text format
    model_cfg: Dict[str, str] = {"model": "", "url": "", "api": ""}
    with open(config_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("model:"):
                model_cfg["model"] = line.split(":", 1)[1].strip()
            elif line.startswith("url:"):
                model_cfg["url"] = line.split(":", 1)[1].strip().strip('"')
            elif line.startswith("api:"):
                model_cfg["api"] = line.split(":", 1)[1].strip().strip('"')

    if not model_cfg["model"] or not model_cfg["url"]:
        raise ValueError("Text config must include model: and url:")

    return model_cfg


# ============================================================================
#  2.  LLM CLIENT
# ============================================================================

class LLMClient:
    """Thin wrapper around OpenAI-compatible chat completions."""

    def __init__(self, config: Dict[str, str], timeout: float = 120.0):
        self.model = config.get("model", "")
        base_url = config.get("url", "")
        if base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")]

        self.client = OpenAI(
            api_key=config.get("api", ""),
            base_url=base_url,
            timeout=timeout,
        )

    def extract_json(
        self,
        system_prompt: str,
        user_prompt: Any,
        max_retries: int = 3,
        timeout: float = 120.0,
    ) -> List[Dict[str, Any]]:
        """Call the LLM and parse response as JSON list of objects."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=str(self.model),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=8192,
                    timeout=timeout,
                )

                content_val = response.choices[0].message.content
                if not content_val:
                    print(f"    WARN empty response on attempt {attempt + 1}", flush=True)
                    if attempt < max_retries - 1:
                        time.sleep(3)
                        continue
                    return []

                content = content_val.strip()

                # Remove markdown code fences when present.
                if content.startswith("```"):
                    first_nl = content.find("\n")
                    if first_nl != -1:
                        content = content[first_nl + 1 :]
                    if content.endswith("```"):
                        content = content[:-3].strip()

                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        return parsed
                    return []
                except json.JSONDecodeError:
                    # Lightweight repair path.
                    last_brace = content.rfind("}")
                    if last_brace != -1:
                        repaired = content[: last_brace + 1] + "\n]"
                        try:
                            parsed = json.loads(repaired)
                            if isinstance(parsed, list):
                                print(f"    INFO recovered {len(parsed)} items via JSON repair")
                                return parsed
                        except Exception:
                            pass

                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        return []

            except Exception as exc:
                print(f"    WARN API error on attempt {attempt + 1}: {exc}", flush=True)
                if attempt < max_retries - 1:
                    time.sleep(4)
                else:
                    return []

        return []

    def solve_question(self, system_prompt: str, question_text: str, max_retries: int = 3) -> Optional[str]:
        """Call LLM to solve one question and return short answer string."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=str(self.model),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": "Solve this question and return only the final answer:\n\n" + question_text,
                        },
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                    timeout=120.0,
                )

                answer = response.choices[0].message.content
                if answer:
                    return answer.strip()

                if attempt < max_retries - 1:
                    time.sleep(3)
            except Exception as exc:
                print(f"    WARN solve API error on attempt {attempt + 1}: {exc}")
                if attempt < max_retries - 1:
                    time.sleep(4)
        return None


# ============================================================================
#  3.  SEGMENTATION
# ============================================================================

BASE_SYSTEM_PROMPT: str = """\
You are an expert at parsing OCR text from Indian competitive exam papers \
(IIT-JEE, NEET, GATE, MHT-CET, WBJEE, VITEEE, BITSAT, etc.).

You will receive the full OCR text (or messy PDF text) of an exam paper. \
Your job is to:
1. {subject_instruction}
2. For each question, extract:
   - The QUESTION: includes the question number, the question text, all \
options (A), (B), (C), (D), and any associated paragraph/comprehension \
text or matrix/column info that belongs to it.
   - The ANSWER: the correct answer option(s) or value(s) that appear \
after the "Answer" marker (or "ANSWER:" etc.).

CRITICAL RULES:
- {subject_critical_rule}
- Preserve all LaTeX formatting exactly as-is.
- Do not modify, simplify, or reformat math expressions.
- If no answer is provided near the question, check for answer key tables \
near the end and map by question number.
- QUESTION must end before Answer marker; do not leak answer text into question.
- For matrix-match questions, include full matrix text and full mapping answer.

Return only a valid JSON array with exactly keys: "question", "answer".
"""


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", "", text).lower()


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n\n".join(str(x) for x in value)
    return str(value)


def _find_first_present_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower_to_original = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]
    return None


class Extractor:
    """Extract question-answer segmentation from OCR parquet/PDF sources."""

    def __init__(self, config_path: str, subjects: str = "all", model_name: Optional[str] = None):
        self.config = read_model_config(config_path, model_name=model_name)
        self.client = LLMClient(self.config)
        self.system_prompt = self._build_system_prompt(subjects)

    @staticmethod
    def _build_system_prompt(subjects: str) -> str:
        if subjects.lower() == "physics":
            return BASE_SYSTEM_PROMPT.format(
                subject_instruction=(
                    "Identify and extract only Physics questions. Ignore non-Physics sections."
                ),
                subject_critical_rule="Only extract Physics questions.",
            )
        return BASE_SYSTEM_PROMPT.format(
            subject_instruction="Identify and extract all questions from all subjects.",
            subject_critical_rule="Extract all questions regardless of subject.",
        )

    def segment_files(
        self,
        input_files: List[Path],
        output_dir: Path,
        max_workers: int = 5,
        resume: bool = True,
    ) -> Path:
        """Segment each input file and save per-file parquet + merged parquet."""
        output_dir.mkdir(parents=True, exist_ok=True)

        segmented_paths: List[Path] = []
        for src_file in input_files:
            suffix = src_file.suffix.lower()
            segmented_path = output_dir / f"{src_file.stem}_segmented.parquet"

            if resume and segmented_path.exists():
                print(f"[resume] skipping {src_file.name} (found {segmented_path.name})")
                segmented_paths.append(segmented_path)
                continue

            print(f"Processing {src_file.name}")
            if suffix == ".parquet":
                records = self._segment_parquet(src_file, max_workers=max_workers)
            elif suffix == ".pdf":
                records = self._segment_pdf(src_file)
            else:
                print(f"  WARN unsupported input type: {src_file}")
                continue

            result_df = pd.DataFrame(records, columns=["source_file", "filename", "question", "answer"])
            if not result_df.empty:
                result_df["norm_q"] = result_df["question"].astype(str).map(_normalize_text)
                result_df = result_df[result_df["norm_q"] != ""]
                result_df = result_df.drop_duplicates(subset=["norm_q"]).drop(columns=["norm_q"])

            result_df.to_parquet(segmented_path, index=False)
            result_df.to_csv(segmented_path.with_suffix(".csv"), index=False)
            segmented_paths.append(segmented_path)
            print(f"  saved {len(result_df)} rows -> {segmented_path}")

        return self._build_merged_output(segmented_paths, output_dir)

    def _build_merged_output(self, segmented_paths: List[Path], output_dir: Path) -> Path:
        if not segmented_paths:
            merged = pd.DataFrame(columns=["source_file", "filename", "question", "answer"])
        else:
            frames: List[pd.DataFrame] = []
            for path in segmented_paths:
                if path.exists():
                    try:
                        frames.append(pd.read_parquet(path))
                    except Exception as exc:
                        print(f"  WARN cannot read {path}: {exc}")
            merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
                columns=["source_file", "filename", "question", "answer"]
            )

        if not merged.empty:
            merged["norm_q"] = merged["question"].astype(str).map(_normalize_text)
            merged = merged[merged["norm_q"] != ""]
            merged = merged.drop_duplicates(subset=["norm_q"]).drop(columns=["norm_q"])

        merged_parquet = output_dir / DEFAULT_MERGED_DATASET_NAME
        merged_csv = merged_parquet.with_suffix(".csv")
        merged.to_parquet(merged_parquet, index=False)
        merged.to_csv(merged_csv, index=False)
        print(f"Merged segmented dataset: {len(merged)} rows -> {merged_parquet}")
        return merged_parquet

    def _segment_parquet(self, parquet_path: Path, max_workers: int) -> List[Dict[str, str]]:
        df = pd.read_parquet(parquet_path)
        if df.empty:
            return []

        filename_col = _find_first_present_column(
            df,
            ["file_name", "filename", "pdf_name", "source", "document", "doc_name"],
        )
        text_col = _find_first_present_column(
            df,
            ["text", "ocr_text", "content", "raw_text", "extracted_text"],
        )

        if text_col is None:
            raise ValueError(
                f"{parquet_path.name} has no text column. Expected one of: "
                "text, ocr_text, content, raw_text, extracted_text"
            )

        results: List[Dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for row_idx, row in df.iterrows():
                future = executor.submit(
                    self._segment_row,
                    row_idx,
                    row,
                    parquet_path.name,
                    filename_col,
                    text_col,
                )
                futures[future] = row_idx

            for fut in as_completed(futures):
                row_idx = futures[fut]
                try:
                    rows = fut.result()
                    if rows:
                        results.extend(rows)
                        print(f"  row {row_idx}: extracted {len(rows)} qa pairs")
                except Exception as exc:
                    print(f"  WARN row {row_idx}: {exc}")

        return results

    def _segment_row(
        self,
        row_idx: int,
        row: Any,
        source_file: str,
        filename_col: Optional[str],
        text_col: str,
    ) -> List[Dict[str, str]]:
        text_val = row[text_col]
        full_text = _to_text(text_val)
        if not full_text.strip():
            return []

        if filename_col and filename_col in row:
            filename = str(row[filename_col])
        else:
            filename = f"{Path(source_file).stem}__row_{row_idx}"

        user_prompt = (
            "Here is the OCR text of an exam document. Segment it into question-answer pairs:\n\n"
            + full_text
        )
        qa_list = self.client.extract_json(self.system_prompt, user_prompt)

        seen: Set[str] = set()
        out: List[Dict[str, str]] = []
        for qa in qa_list:
            q_text = str(qa.get("question", "")).strip()
            a_text = str(qa.get("answer", "")).strip()
            norm_q = _normalize_text(q_text)
            if norm_q and norm_q not in seen:
                seen.add(norm_q)
                out.append(
                    {
                        "source_file": source_file,
                        "filename": filename,
                        "question": q_text,
                        "answer": a_text,
                    }
                )
        return out

    def _segment_pdf(self, pdf_path: Path) -> List[Dict[str, str]]:
        if fitz is None:
            raise ImportError("PyMuPDF is required for PDF input. Install: pip install PyMuPDF")

        filename = pdf_path.name
        try:
            doc = fitz.open(str(pdf_path))  # type: ignore[arg-type]
            pages = [page.get_text() for page in doc]
            full_text = "\n\n".join(pages)
        except Exception as exc:
            print(f"  WARN cannot parse PDF {filename}: {exc}")
            return []

        # Fallback to image mode for scanned PDFs with little extracted text.
        if len(full_text.strip()) < 100:
            user_content: List[Dict[str, Any]] = [
                {
                    "type": "text",
                    "text": (
                        "Here are scanned pages of an exam paper. Segment into question-answer pairs. "
                        "Check end pages for answer keys."
                    ),
                }
            ]
            try:
                doc = fitz.open(str(pdf_path))  # type: ignore[arg-type]
                for page in doc:
                    pix = page.get_pixmap(dpi=150)
                    img_data = pix.tobytes("jpeg")
                    b64_img = base64.b64encode(img_data).decode("utf-8")
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                        }
                    )
                qa_list = self.client.extract_json(self.system_prompt, user_content)
            except Exception:
                qa_list = []
        else:
            user_prompt = (
                "Here is extracted text from an exam paper. Segment into question-answer pairs:\n\n"
                + full_text
            )
            qa_list = self.client.extract_json(self.system_prompt, user_prompt)

        seen: Set[str] = set()
        out: List[Dict[str, str]] = []
        for qa in qa_list:
            q_text = str(qa.get("question", "")).strip()
            a_text = str(qa.get("answer", "")).strip()
            norm_q = _normalize_text(q_text)
            if norm_q and norm_q not in seen:
                seen.add(norm_q)
                out.append(
                    {
                        "source_file": filename,
                        "filename": filename,
                        "question": q_text,
                        "answer": a_text,
                    }
                )
        return out


# ============================================================================
#  4.  CLEANING
# ============================================================================


def _is_weird(answer: Any) -> bool:
    if pd.isna(answer) or not isinstance(answer, str):
        return True

    ans_clean = str(answer).strip()
    if not ans_clean:
        return False

    if (
        len(re.findall(r"\d\s+\d", ans_clean)) >= 3
        and len(re.sub(r"[\d\s\(\)]", "", ans_clean)) == 0
    ):
        return True

    if re.search(r"\d \d \d \d \d", ans_clean):
        return True

    if (
        re.search(r"^[\d\s\(\)]+$", ans_clean)
        and len(ans_clean) > 8
        and ans_clean.count(" ") > 3
    ):
        return True

    lower_ans = ans_clean.lower()
    if "q and r" in lower_ans and (
        "a:" in lower_ans or "**a:" in lower_ans or "b:" in lower_ans
    ):
        return True

    if len(re.findall(r"\b[A-Da-d]\s*:", ans_clean)) >= 2:
        return True

    if len(re.findall(r"\*\*[A-Da-d]\*\*\s*:", ans_clean)) >= 2:
        return True

    return False


def _clean_answer_text(ans: Any) -> str:
    if pd.isna(ans) or not ans:
        return ""

    ans_str = str(ans).strip()
    if len(ans_str) < 20 and "\n" not in ans_str:
        return ans_str

    match = re.search(
        r"(?i)answer\s*is\s*[:=]?\s*([A-Za-z0-9\(\),\-\;\.\s]{1,20})$",
        ans_str,
    )
    if match:
        return match.group(1).strip()

    if ")" in ans_str:
        parts = re.findall(r"\([A-D]\)", ans_str)
        if parts:
            return "".join(parts[-4:]).strip()

    lines = [line.strip() for line in ans_str.split("\n") if line.strip()]
    if lines and len(lines[-1]) < 20:
        return lines[-1]

    return ans_str


def clean_dataset(parquet_path: Path) -> None:
    """Remove malformed answers and trim verbose ones in-place."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"Original rows: {len(df)}")

    weird_mask = df["answer"].apply(_is_weird)
    weird_count = int(weird_mask.sum())
    print(f"Filtering out {weird_count} malformed/weird answers")
    df = df[~weird_mask]

    df["clean_answer"] = df["answer"].apply(_clean_answer_text)
    changed = int((df["answer"] != df["clean_answer"]).sum())
    print(f"Cleaned {changed} verbose answers")

    df["answer"] = df["clean_answer"]
    df = df.drop(columns=["clean_answer"])

    df.to_parquet(parquet_path, index=False)
    df.to_csv(parquet_path.with_suffix(".csv"), index=False)
    print(f"Saved cleaned dataset: {parquet_path}")


# ============================================================================
#  5.  SOLVE EMPTY ANSWERS
# ============================================================================

SOLVER_PROMPT: str = """\
You are an expert at solving Indian competitive exam problems
(IIT-JEE Advanced, JEE Mains, NEET, GATE, etc.).

You will be given one question. Return only the final answer.

Rules:
- single-choice MCQ -> (C)
- multi-choice MCQ  -> (A)(B)(D)
- numeric/integer   -> only the number
- matrix match      -> A-p,q; B-r; C-s,t; D-p
- no explanation, no extra text
"""


def solve_empty_answers(
    config_path: str,
    dataset_path: Path,
    max_workers: int = 50,
    model_name: Optional[str] = None,
) -> None:
    """Use LLM to fill empty answers in-place."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    config = read_model_config(config_path, model_name=model_name)
    client = LLMClient(config, timeout=120.0)

    df = pd.read_parquet(dataset_path)
    df["answer"] = df["answer"].fillna("").astype(str)
    empty_mask = df["answer"].str.strip() == ""
    empty_count = int(empty_mask.sum())

    print(f"Empty answers found: {empty_count} / {len(df)}")
    if empty_count == 0:
        print("No empty answers to solve")
        return

    empty_indices: List[int] = df[empty_mask].index.tolist()
    print(f"Solving empty answers with workers={max_workers}")

    results: Dict[int, str] = {}
    solved = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(client.solve_question, SOLVER_PROMPT, str(df.at[idx, "question"])): idx
            for idx in empty_indices
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                answer = future.result()
                if answer:
                    results[idx] = answer
                    solved += 1
                    print(f"  solved {solved + failed}/{empty_count}: {answer[:60]}")
                else:
                    failed += 1
                    print(f"  failed {solved + failed}/{empty_count}")
            except Exception as exc:
                failed += 1
                print(f"  failed {solved + failed}/{empty_count}: {exc}")

    for idx, ans in results.items():
        df.at[idx, "answer"] = ans

    df.to_parquet(dataset_path, index=False)
    df.to_csv(dataset_path.with_suffix(".csv"), index=False)

    remaining_empty = int((df["answer"].str.strip() == "").sum())
    print(f"Solved: {solved}/{empty_count}")
    print(f"Remaining empty: {remaining_empty}")


# ============================================================================
#  6.  CLI
# ============================================================================


def _resolve_input_files(input_file: Optional[str], input_dir: str, mode: str) -> List[Path]:
    if input_file:
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return [path]

    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")

    files: List[Path] = []
    if mode in ("auto", "parquet"):
        files.extend(Path(p) for p in glob.glob(str(root / "*.parquet")))
    if mode in ("auto", "pdf"):
        files.extend(Path(p) for p in glob.glob(str(root / "*.pdf")))

    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No input files found in {root} for mode={mode}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step-2 segmentation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    python segmentation_pipeline.py --step segment
    python segmentation_pipeline.py --step segment --input OCR_output/JA_maths_ocr.parquet
    python segmentation_pipeline.py --step segment --mode parquet --workers 8
  python segmentation_pipeline.py --step segment --subjects physics --workers 8
""",
    )

    parser.add_argument(
        "--step",
        choices=["segment", "extract"],
        default="segment",
        help="Pipeline step. 'extract' is accepted as alias of 'segment'.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to model config file")
    parser.add_argument("--model-name", default=None, help="Model name selector for JSON config lists")

    parser.add_argument("--input", default=None, help="Single input file (.parquet/.pdf)")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Input directory")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")

    parser.add_argument("--mode", choices=["auto", "parquet", "pdf"], default="auto")
    parser.add_argument("--subjects", choices=["all", "physics"], default="all")
    parser.add_argument("--workers", type=int, default=5)

    parser.set_defaults(resume=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Do not skip existing segmented files")

    args = parser.parse_args()

    step = "segment" if args.step == "extract" else args.step
    output_dir = Path(args.output_dir)
    if step != "segment":
        raise ValueError("Unsupported step")

    print("=" * 60)
    print("STEP 2: SEGMENTATION")
    print("=" * 60)
    input_files = _resolve_input_files(args.input, args.input_dir, args.mode)
    print(f"Found {len(input_files)} input file(s)")

    extractor = Extractor(args.config, subjects=args.subjects, model_name=args.model_name)
    merged_dataset = extractor.segment_files(
        input_files,
        output_dir=output_dir,
        max_workers=args.workers,
        resume=args.resume,
    )

    print(f"Segmentation finished. Merged output: {merged_dataset}")


if __name__ == "__main__":
    main()
