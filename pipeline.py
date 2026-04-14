#!/usr/bin/env python3
"""
Generalized CoT Pipeline
=========================
Reads any CSV/Parquet from Datasets/, auto-maps columns via LLM,
runs CoT generation with rejection sampling, and produces
standardized output.

Usage:
  python pipeline.py Datasets/data.csv
  python pipeline.py Datasets/data.csv --config config.yaml
  python pipeline.py Datasets/data.csv --column-map question=Q answer="gold answer"
  python pipeline.py Datasets/data.csv --dry-run
  python pipeline.py Datasets/data.csv --workers 12
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = Path(__file__).resolve().parent

# ════════════════════════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cot_pipeline")


# ════════════════════════════════════════════════════════════════
#  CONFIGURATION  (typed dataclasses replace raw dicts)
# ════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    url: str
    model: str
    name: str = ""
    api_key: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.model.replace("/", "_")


@dataclass
class PipelineConfig:
    output_dir: Path = BASE_DIR / "Generated_Dataset"
    max_attempts: int = 3
    temperature: float = 0.2
    retry_temperature: float = 0.6
    max_tokens: int = 8000
    timeout_s: int = 180
    max_workers: int = 8
    made_by: str = "team 3"
    generation_type: str = "reasoning"
    known_domains: List[str] = field(default_factory=lambda: [
        "Chemistry", "Math", "Physics", "Biology", "Language",
    ])

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)


def load_config(path: str) -> Tuple[ModelConfig, ModelConfig, PipelineConfig]:
    """Load YAML config → typed dataclass instances.

    Returns (mapper_cfg, model_cfg, pipeline_cfg).
    """
    with open(path) as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    def _model(section: str) -> ModelConfig:
        d = raw.get(section, {})
        return ModelConfig(
            url=d.get("url", ""),
            model=d.get("model", ""),
            name=d.get("name", ""),
            api_key=d.get("api_key", ""),
        )

    pipe_raw = raw.get("pipeline", {})
    pipeline = PipelineConfig(
        output_dir=pipe_raw.get("output_dir", str(BASE_DIR / "Generated_Dataset")),
        max_attempts=pipe_raw.get("max_attempts", 3),
        temperature=pipe_raw.get("temperature", 0.2),
        retry_temperature=pipe_raw.get("retry_temperature", 0.6),
        max_tokens=pipe_raw.get("max_tokens", 8000),
        timeout_s=pipe_raw.get("timeout_s", 180),
        max_workers=pipe_raw.get("max_workers", 8),
        made_by=pipe_raw.get("made_by", "team 3"),
        generation_type=pipe_raw.get("generation_type", "reasoning"),
        known_domains=raw.get("known_domains", PipelineConfig().known_domains),
    )
    return _model("mapper_model"), _model("model"), pipeline


# ════════════════════════════════════════════════════════════════
#  TYPED RECORD  (replaces Dict[str, Any] in hot path)
# ════════════════════════════════════════════════════════════════

class QuestionRecord(TypedDict, total=False):
    _row_idx: int
    question: str
    content: str
    reasoning_content: str
    model_answer: str
    ground_truth: str
    is_correct: bool
    attempts: int
    error: str


# ════════════════════════════════════════════════════════════════
#  HTTP  –  per-thread session with connection pooling
# ════════════════════════════════════════════════════════════════

_local = threading.local()


def _get_session() -> requests.Session:
    """Return a per-thread Session with retry/keep-alive configured."""
    if not getattr(_local, "session", None):
        session = requests.Session()
        retry = Retry(
            total=0,           # We handle retries ourselves
            backoff_factor=0,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            pool_connections=1,
            pool_maxsize=4,
            max_retries=retry,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _local.session = session
    return _local.session


def _llm_post(
    url: str,
    payload: Dict[str, Any],
    api_key: str = "",
    timeout_s: int = 30,
    max_retries: int = 3,
) -> str:
    """POST to an OpenAI-compatible endpoint; return the assistant message text.

    Uses a thread-local Session for connection reuse, and exponential backoff
    on transient failures.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    session = _get_session()
    last_exc: Exception = RuntimeError("no attempts made")

    for attempt in range(max_retries):
        try:
            resp = session.post(url, headers=headers, json=payload, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            return (
                data["choices"][0]["message"].get("content", "") or ""
            ).strip()
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + 0.5)

    raise last_exc


def _llm_complete(
    url: str,
    model: str,
    prompt: str,
    *,
    api_key: str = "",
    temperature: float = 0.0,
    max_tokens: int = 500,
    timeout_s: int = 30,
    max_retries: int = 3,
) -> str:
    """Single-turn LLM completion."""
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return _llm_post(url, payload, api_key=api_key, timeout_s=timeout_s,
                     max_retries=max_retries)


# ════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════

def load_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    ext = p.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(p)
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported file format: {ext}  (only .csv / .parquet)")

    log.info("Loaded %d rows × %d columns from %s", len(df), len(df.columns), p.name)
    log.info("Columns: %s", list(df.columns))
    return df


# ════════════════════════════════════════════════════════════════
#  COLUMN MAPPING
# ════════════════════════════════════════════════════════════════

REQUIRED_MAPPING: Dict[str, str] = {
    "question": "The column containing the question / problem text",
    "ground_truth": "Correct / gold answer column (or answer)",
}
OPTIONAL_MAPPING: Dict[str, str] = {
    "domain":       "Subject domain",
    "task":         "Task / question type",
    "source":       "Source or origin of the dataset",
    "language":     "Language of the question",
}

_COLUMN_MAPPING_PROMPT = """\
You are a data-schema analyst. Map dataset columns to canonical pipeline columns.

REQUIRED: question, ground_truth (ground_truth can come from an answer column)
OPTIONAL: domain, task, source, language

Return ONLY a valid JSON object: {{"question": "actual_col", ...}}
Omit any canonical column with no match. Do NOT invent columns.

DATASET COLUMNS: {columns}

SAMPLE ROWS:
{samples}

JSON mapping:"""


def _canonical_mapping_key(raw_key: str) -> str:
    key = str(raw_key or "").strip().lower()
    if key == "answer":
        return "ground_truth"
    return key


def map_columns_with_llm(
    df: pd.DataFrame,
    cfg: ModelConfig,
) -> Dict[str, str]:
    prompt = _COLUMN_MAPPING_PROMPT.format(
        columns=list(df.columns),
        samples=df.head(2).to_string(index=False),
    )
    log.info("Asking LLM to identify column mapping …")
    raw = _llm_complete(cfg.url, cfg.model, prompt, api_key=cfg.api_key)

    m = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if not m:
        log.warning("LLM returned non-JSON response:\n%s", raw)
        return {}

    try:
        mapping: Dict[str, str] = json.loads(m.group(0))
    except json.JSONDecodeError:
        log.warning("Failed to parse LLM JSON:\n%s", raw)
        return {}

    valid: Dict[str, str] = {}
    allowed = set(REQUIRED_MAPPING) | set(OPTIONAL_MAPPING)
    for canonical, actual in mapping.items():
        canonical = _canonical_mapping_key(canonical)
        if canonical not in allowed:
            continue
        if actual in df.columns:
            valid[canonical] = actual
        else:
            log.warning("LLM mapped '%s' → '%s' but column not found, skipping",
                        canonical, actual)
    return valid


_FUZZY_ALIASES: Dict[str, List[str]] = {
    "question":     ["question", "problem", "question_text", "problem_text", "q"],
    "ground_truth": ["ground_truth", "gold_answer", "gold answer", "answer",
                     "correct_answer", "correct_options", "expected_answer", "gold", "gt",
                     "target", "solution"],
    "domain":       ["domain", "subject", "sub", "category", "topic", "discipline"],
    "task":         ["task", "type", "question_type", "qtype", "q_type", "format"],
    "source":       ["source", "origin", "dataset", "exam", "paper", "competition"],
    "language":     ["language", "lang", "locale", "medium"],
}


def fuzzy_match_columns(df: pd.DataFrame) -> Dict[str, str]:
    lower = {c.lower().strip(): c for c in df.columns}
    mapping: Dict[str, str] = {}
    for canonical, aliases in _FUZZY_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lower:
                mapping[canonical] = lower[alias.lower()]
                break
    return mapping


def confirm_mapping(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    *,
    interactive: bool = False,
) -> Dict[str, str]:
    sep = "=" * 55
    log.info(sep)
    log.info("  Detected Column Mapping")
    log.info(sep)
    for key in list(REQUIRED_MAPPING) + list(OPTIONAL_MAPPING):
        label = "(required)" if key in REQUIRED_MAPPING else "(optional)"
        mapped = mapping.get(key, "---")
        log.info("  %-15s ← '%s'  %s", key, mapped, label)
    log.info("  Available columns: %s", list(df.columns))

    missing = [k for k in REQUIRED_MAPPING if k not in mapping]
    if missing:
        log.warning("Missing REQUIRED mappings: %s", missing)

    if not interactive:
        log.info("  ✓ Auto-confirmed")
        return mapping

    choice = input("  Confirm? [Y/edit/q]: ").strip().lower()
    if choice in ("q", "quit", "exit"):
        sys.exit(0)
    if choice in ("", "y", "yes"):
        return mapping

    edited = dict(mapping)
    for key in list(REQUIRED_MAPPING) + list(OPTIONAL_MAPPING):
        label = "(required)" if key in REQUIRED_MAPPING else "(optional)"
        current = edited.get(key, "")
        val = input(f"    {key} [{current}] {label}: ").strip()
        if val == "-":
            edited.pop(key, None)
        elif val:
            if val in df.columns:
                edited[key] = val
            else:
                log.warning("Column '%s' not found, keeping previous", val)
    return edited


# ════════════════════════════════════════════════════════════════
#  LLM METADATA ENRICHMENT
# ════════════════════════════════════════════════════════════════

_ENRICH_PROMPT = """\
Classify this question.

1. domain: Chemistry | Math | Physics | Biology | Language
2. task: QA | mcq_single | mcq_multiple | numerical_integer | matching_columns | paragraph | code_generation
3. language: English | Hindi | Other

Return ONLY JSON: {{"domain": "...", "task": "...", "language": "..."}}. Use null if unknown.

QUESTION: {question}
ANSWER:   {answer}

JSON:"""


def _enrich_one(
    idx: Any,
    row: Dict[str, Any],
    fields: List[str],
    cfg: ModelConfig,
) -> Tuple[Any, Dict[str, str]]:
    question = str(row.get("question", ""))[:500]
    answer = str(row.get("ground_truth", ""))
    prompt = _ENRICH_PROMPT.format(question=question, answer=answer)
    try:
        raw = _llm_complete(cfg.url, cfg.model, prompt, api_key=cfg.api_key)
        m = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if not m:
            return idx, {}
        result: Dict[str, Any] = json.loads(m.group(0))
        updates: Dict[str, str] = {}
        for f in fields:
            current = str(row.get(f, "")).strip().lower()
            if current in ("", "nan", "none", "unknown", "null"):
                new_val = result.get(f)
                if new_val and str(new_val).lower() != "null":
                    updates[f] = str(new_val)
        return idx, updates
    except Exception as exc:
        return idx, {"_error": str(exc)}


def enrich_metadata(
    df: pd.DataFrame,
    fields: List[str],
    cfg: ModelConfig,
    max_workers: int = 8,
) -> pd.DataFrame:
    if not fields:
        return df

    _is_blank = lambda v: str(v).strip().lower() in ("", "nan", "none", "unknown", "null")  # noqa: E731
    to_enrich = [
        (idx, row.to_dict())
        for idx, row in df.iterrows()
        if any(_is_blank(row.get(f, "")) for f in fields)
    ]

    if not to_enrich:
        log.info("No rows need enrichment")
        return df

    log.info("Enriching %d/%d rows  fields=%s  workers=%d",
             len(to_enrich), len(df), fields, max_workers)

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_enrich_one, idx, row, fields, cfg): idx
                for idx, row in to_enrich}
        for fut in as_completed(futs):
            idx, updates = fut.result()
            if "_error" in updates:
                log.warning("Row %s enrichment failed: %s", idx, updates["_error"])
            else:
                for f, v in updates.items():
                    df.at[idx, f] = v
            done += 1
            if done % 50 == 0 or done == len(to_enrich):
                log.info("  … %d/%d enriched", done, len(to_enrich))

    counts = {
        f: int(df[f].apply(lambda x: not _is_blank(x)).sum())   # type: ignore[arg-type]
        for f in fields
    }
    log.info("Enrichment done: %s", counts)
    return df


# ════════════════════════════════════════════════════════════════
#  WORKING DATAFRAME PREPARATION
# ════════════════════════════════════════════════════════════════

def _infer_source_from_filename(stem: str) -> str:
    m = re.search(r"(JEE[- _]?(?:ADV(?:ANCED)?|MAIN[S]?)|NEET|AIEEE)",
                  stem, re.IGNORECASE)
    return m.group(1).upper().replace("_", "-") if m else ""


_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "Math":           ["math","algebra","calculus","geometry","probability",
                       "vector","trigonometry","matrix","derivative","integral","equation"],
    "Physics":        ["physics","thermo","electric","magnetic","kinetic","optics",
                       "mechanics","force","velocity","momentum","energy","quantum","lens"],
    "Chemistry":      ["chemistry","molecule","atom","reaction","acid","base","organic",
                       "inorganic","polymer","bond","equilibrium"],
    "Biology":        ["biology","cell","genetics","evolution","dna","rna","protein",
                       "enzyme","ecology","botany","zoology","photosynthesis","respiration"],
    "Language":       ["grammar","vocabulary","comprehension","noun","verb","adjective",
                       "sentence","paragraph","essay","synonym","antonym"],
}


def canonicalize_domain_label(raw: Any) -> str:
    v = re.sub(r"[^a-z]+", " ", str(raw or "").strip().lower())
    if not v.strip() or v.strip() in ("nan", "none", "null", "unknown"):
        return "unknown"
    if "math" in v:
        return "Math"
    if "phy" in v:
        return "Physics"
    if "chem" in v:
        return "Chemistry"
    if "bio" in v:
        return "Biology"
    if "lang" in v or "english" in v or "hindi" in v:
        return "Language"
    return str(raw).strip().title()


def _infer_domain(question: str, source: str, current: str) -> str:
    current_domain = canonicalize_domain_label(current)
    if current_domain != "unknown":
        return current_domain
    text = f"{source} {question[:500]}".lower()
    best, best_score = "unknown", 0
    for dom, kws in _DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in text)
        if score > best_score:
            best, best_score = dom, score
    return best if best_score > 0 else "unknown"


_BLANK = {"", "nan", "none", "null"}


def prepare_working_df(
    df: pd.DataFrame,
    column_map: Dict[str, str],
    input_filename: str = "",
) -> pd.DataFrame:
    working = pd.DataFrame()
    working["_row_idx"] = range(len(df))

    for canonical, actual in column_map.items():
        if actual in df.columns:
            working[canonical] = df[actual].values

    if "question" not in working.columns:
        raise ValueError("Column map must include 'question'")

    for col in ("source", "domain", "task", "ground_truth", "language"):
        if col not in working.columns:
            working[col] = ""

    # Source is standardized to input filename for the whole dataset.
    if input_filename:
        working["source"] = Path(input_filename).name

    working["domain"] = [
        _infer_domain(q, s, d)
        for q, s, d in zip(
            working["question"].astype(str),
            working["source"].astype(str),
            working["domain"].astype(str),
        )
    ]

    # Normalize blanks
    src = working["source"].astype(str).str.strip()
    working["source"] = src.mask(src.str.lower().isin(_BLANK), "unknown")

    tsk = working["task"].astype(str).str.strip()
    working["task"] = tsk.mask(tsk.str.lower().isin(_BLANK), "QA")

    lang = working["language"].astype(str).str.strip()
    working["language"] = lang.mask(lang.str.lower().isin(_BLANK), "English")

    return working


def clean_question_text(df: pd.DataFrame) -> pd.DataFrame:
    df["question"] = (
        df["question"].astype(str)
        .str.replace(r"\\textbf\{Q\.\d+\}", "", regex=True)
        .str.replace(r"Q\.\d+\s*", "", regex=True)
        .str.replace(r"\\textbf\b\s*", "", regex=True)
        .str.replace(r"\\item\s*", "", regex=True)
        .str.strip()
    )
    return df


# ════════════════════════════════════════════════════════════════
#  ANSWER EXTRACTION
# ════════════════════════════════════════════════════════════════

_WRAPPERS = [
    ("**", "**"), ("__", "__"), ("`", "`"), ("$", "$"),
    ("\\[", "\\]"), ("\\(", "\\)"),
    ("[", "]"),   ("(", ")"),   ("{", "}"), ('"', '"'), ("'", "'"),
]


def _normalize_answer_text(value: str) -> str:
    v = (value or "").strip()
    if not v:
        return ""
    # Take first non-empty line
    v = next((ln.strip() for ln in v.splitlines() if ln.strip()), v)

    prev = None
    while v and v != prev:
        prev = v
        v = v.replace("\\,", "").replace("\\;", "").strip()
        v = v.strip(" \t\n\r,;")
        v = re.sub(r"^\s*(?:\*\*|__)?(?:final\s*answer|answer|ans|correct\s+options?)\s*(?:is|[:\-=])?\s*(?:\*\*|__)?\s*(?:is|[:\-=])?\s*",
                   "", v, flags=re.IGNORECASE).strip()
        boxed = re.fullmatch(r"\\boxed\s*\{(.+)\}", v, flags=re.DOTALL)
        if boxed:
            v = boxed.group(1).strip()
            continue
        latex_text = re.fullmatch(r"\\(?:text|textbf|mathrm)\s*\{(.+)\}", v, flags=re.DOTALL)
        if latex_text:
            v = latex_text.group(1).strip()
            continue
        for left, right in _WRAPPERS:
            if len(v) > len(left) + len(right) and v.startswith(left) and v.endswith(right):
                v = v[len(left): len(v) - len(right)].strip()
                break

    # Normalise MCQ option lists
    if re.fullmatch(r"[A-Da-d](?:\s*,\s*[A-Da-d])+", v):
        return ",".join(p.strip().upper() for p in v.split(","))
    if re.fullmatch(r"[A-Da-d]", v):
        return v.upper()

    v = re.sub(r"\s+", " ", v).strip()
    v = re.sub(r"\s*[;:]\s*$", "", v)
    v = re.sub(r"(?<=[A-Za-z\]\)])\s*[.,]\s*$", "", v)
    return v


_FINAL_ANSWER_PATS = [
    re.compile(r"(?is)\[\s*final\s*answer\s*:\s*(.*?)\s*\]"),
    re.compile(r"(?im)^\s*[\*_]*final\s*answer[\*_]*\s*(?:is|[:\-=])*[\*_]*\s*(.+?)\s*$"),
    re.compile(r"(?im)^\s*[\*_]*(?:final\s*)?answer[\*_]*\s*(?:is|[:\-=])*[\*_]*\s*(.+?)\s*$"),
    re.compile(r"(?im)^\s*[\*_]*answer[\*_]*\s*(?:is|[:\-=])*[\*_]*\s*(.+?)\s*$"),
    re.compile(r"(?im)^\s*[\*_]*ans[\*_]*\s*(?:is|[:\-=])*[\*_]*\s*(.+?)\s*$"),
]
_BOXED_RE = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})+)\}")
_NUMERIC_RE = re.compile(
    r"(?:"
    r"\\frac\s*\{[-+]?\d+\}\s*\{[-+]?\d+\}"
    r"|[-+]?\d+\s*/\s*[-+]?\d+"
    r"|[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
    r"|[-+]?\.\d+(?:[eE][-+]?\d+)?"
    r")"
)


def extract_model_answer(text: str) -> str:
    src = (text or "").replace("\r\n", "\n").strip()
    if not src:
        return ""

    for pat in _FINAL_ANSWER_PATS:
        for candidate in reversed(pat.findall(src)):
            norm = _normalize_answer_text(candidate)
            if norm:
                return norm

    boxed = _BOXED_RE.findall(src)
    if boxed:
        norm = _normalize_answer_text(boxed[-1])
        if norm:
            return norm

    lines = [ln.strip() for ln in src.splitlines() if ln.strip()]
    if not lines:
        return ""

    last = _normalize_answer_text(lines[-1])
    if not last:
        return ""
    if re.fullmatch(r"[A-Da-d](?:\s*,\s*[A-Da-d])*", last):
        return ",".join(p.strip().upper() for p in last.split(",")) if "," in last else last.upper()
    if re.fullmatch(_NUMERIC_RE.pattern, last):
        return last.replace(" ", "")
    return last


# ════════════════════════════════════════════════════════════════
#  ANSWER VERIFICATION
# ════════════════════════════════════════════════════════════════

_TASK_ALIASES: Dict[str, str] = {
    "mcq": "mcq_single",           "single": "mcq_single",
    "single_correct": "mcq_single","mcqsingle": "mcq_single",
    "multiple": "mcq_multiple",    "multiple_correct": "mcq_multiple",
    "mcqmultiple": "mcq_multiple",
    "integer": "numerical_integer","nat": "numerical_integer",
    "numerical": "numerical_integer",
    "range": "numerical_range",    "decimal_range": "numerical_range",
    "float_range": "numerical_range",
    "matching": "matching_columns",
    "paragraph": "paragraph",      "unknown": "unknown",
}


def canonicalize_task(raw: Any) -> str:
    qt = str(raw or "").strip().lower()
    if qt in ("", "nan", "none", "null"):
        return "unknown"
    return _TASK_ALIASES.get(qt, qt)


_MCQ_SEP_RE = re.compile(r"[\s,./\\|&+()\[\]{}<>:;_\"'`-]+")
_OPTION_MARK_RE = re.compile(r"(?i)(?<![A-Za-z0-9])([A-D])\s*[\).:\-]\s*")


def _mcq_letters(text: str) -> str:
    normalized = re.sub(r"\b(?:and|or)\b", ",", str(text or ""), flags=re.IGNORECASE)
    core = _MCQ_SEP_RE.sub("", normalized)
    return ",".join(sorted({c.upper() for c in core if c.upper() in "ABCD"}))


def _is_mcq_like(text: str) -> bool:
    normalized = re.sub(r"\b(?:and|or)\b", ",", str(text or ""), flags=re.IGNORECASE)
    core = _MCQ_SEP_RE.sub("", normalized)
    return bool(core) and bool(re.fullmatch(r"[A-Da-d]+", core))


def _extract_mcq_key(text: str) -> str:
    src = str(text or "").strip()
    if not src:
        return ""

    # Handle JSON list style answers like ["A"] or ["A", "C"].
    try:
        parsed = json.loads(src)
        if isinstance(parsed, list):
            joined = ",".join(str(x) for x in parsed)
            if _is_mcq_like(joined):
                return _mcq_letters(joined)
    except Exception:
        pass

    candidates = [src]
    normalized = _normalize_answer_text(src)
    if normalized and normalized != src:
        candidates.append(normalized)

    for candidate in candidates:
        # Handle LaTeX text wrappers such as \text{C} and \text{Option B}.
        text_letter = re.search(
            r"\\(?:text|textbf|mathrm)\s*\{[^{}]*?(?<![A-Za-z0-9])([A-Da-d])(?![A-Za-z0-9])[^{}]*\}",
            candidate,
        )
        if text_letter:
            return text_letter.group(1).upper()

        # Handle LaTeX boxed format with nested braces: \boxed{C}, \boxed{\text{C}}, etc.
        boxed_match = re.search(r"\\boxed\s*\{", candidate)
        if boxed_match:
            start = boxed_match.end()
            depth = 1
            end = start
            while end < len(candidate) and depth > 0:
                if candidate[end] == '{':
                    depth += 1
                elif candidate[end] == '}':
                    depth -= 1
                end += 1
            if depth == 0:
                boxed_content = candidate[start:end - 1].strip()
                letter_match = re.search(r"(?<![A-Za-z0-9])([A-Da-d])(?![A-Za-z0-9])", boxed_content)
                if letter_match:
                    return letter_match.group(1).upper()
                if _is_mcq_like(boxed_content):
                    result = _mcq_letters(boxed_content)
                    if result:
                        return result

        # Handle LaTeX bracket and parenthesis formats: \[A\] and \(A\)
        latex_bracket = re.findall(r"\\\[\s*([A-Da-d])\s*\\\]", candidate)
        if latex_bracket:
            return latex_bracket[-1].upper()

        paren_matches = re.findall(r"\\\(\s*([A-Da-d])\s*\\\)", candidate)
        if paren_matches:
            return paren_matches[-1].upper()

        # Extract from text patterns such as "correct option is D" and "choice: B".
        option_pattern = re.search(
            r"(?:correct\s+(?:option|choice|answer)|final\s*answer|answer|ans|option|choice)\s*(?:is|=|:)?\s*(?:option|choice)?\s*\(?\s*([A-Da-d])\s*\)?",
            candidate,
            re.IGNORECASE,
        )
        if option_pattern:
            return option_pattern.group(1).upper()

        for bracketed in reversed(re.findall(r"\[([^\[\]]+)\]", candidate)):
            if _is_mcq_like(bracketed):
                return _mcq_letters(bracketed)

        if _is_mcq_like(candidate):
            return _mcq_letters(candidate)

    return ""


def _extract_option_map(question: str) -> Dict[str, str]:
    """Extract option text keyed by A/B/C/D from a question string."""
    raw = html.unescape(str(question or ""))
    if not raw.strip():
        return {}

    # Remove HTML tags and normalize whitespace.
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text).strip()

    # Focus on the options region when present.
    m = re.search(r"\boptions?\b\s*:?", text, flags=re.IGNORECASE)
    block = text[m.end():].strip() if m else text

    matches = list(_OPTION_MARK_RE.finditer(block))
    if len(matches) < 2:
        return {}

    option_map: Dict[str, str] = {}
    for i, match in enumerate(matches):
        letter = match.group(1).upper()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        value = re.sub(r"\s+", " ", block[start:end]).strip()
        if value:
            option_map[letter] = value

    # Keep only canonical MCQ keys.
    return {k: option_map[k] for k in ("A", "B", "C", "D") if k in option_map}


def _answers_equivalent(lhs: str, rhs: str, tol: float = 1e-6) -> bool:
    """Compare answers using numeric and normalized-text equivalence."""
    if _numeric_match(lhs, rhs, tol=tol):
        return True
    nl = _normalize_string(_normalize_answer_text(lhs))
    nr = _normalize_string(_normalize_answer_text(rhs))
    return bool(nl and nr and nl == nr)


def _split_answer_pieces(text: str) -> List[str]:
    normalized = re.sub(r"\b(?:and|or)\b", ",", str(text or ""), flags=re.IGNORECASE)
    return [p.strip() for p in re.split(r",|;|\|", normalized) if p.strip()]


def _resolve_mcq_answer(question: str, answer: str, numeric_tol: float = 1e-6) -> str:
    """Resolve model/gold answer text to canonical MCQ labels when possible."""
    ans = _normalize_answer_text(str(answer or ""))
    if not ans:
        return ""

    direct = _extract_mcq_key(ans)
    if direct:
        return direct

    option_map = _extract_option_map(question)
    if not option_map:
        return ""

    pieces = _split_answer_pieces(ans)
    if not pieces:
        pieces = [ans]

    matched_letters: List[str] = []
    used_letters: set[str] = set()
    for part in pieces:
        found = None
        for letter, value in option_map.items():
            if letter in used_letters:
                continue
            if _answers_equivalent(part, value, tol=numeric_tol):
                found = letter
                break
        if found is None:
            return ""
        used_letters.add(found)
        matched_letters.append(found)

    return ",".join(sorted(set(matched_letters)))


def _normalize_mcq_model_answer(
    question: str,
    model_answer: str,
    task: str,
    ground_truth: str = "",
    numeric_tol: float = 1e-6,
) -> str:
    """Map MCQ option values back to labels (A/B/C/D) when possible."""
    ma = str(model_answer or "").strip()
    if not ma:
        return ""

    qt = canonicalize_task(task)
    gt_mcq = _extract_mcq_key(ground_truth)
    if qt not in ("mcq_single", "mcq_multiple") and not gt_mcq:
        return ma

    resolved = _resolve_mcq_answer(question, ma, numeric_tol=numeric_tol)
    return resolved or ma


def _normalize_string(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower().strip().strip(".,;:\"'"))


def _match_pairs(text: str) -> set[str]:
    return {
        f"{num}-{char.upper()}"
        for num, char in re.findall(r"(\d+)\s*[-–]\s*([A-Za-z])", str(text or ""))
    }


def _parse_numeric(token: str) -> float:
    t = token.strip().replace(",", "")
    if "\\frac" in t:
        m = re.search(r"\\frac\s*\{([-+]?\d+)\}\s*\{([-+]?\d+)\}", t)
        if m:
            num, den = int(m.group(1)), int(m.group(2))
            if den == 0:
                raise ValueError("Denominator is zero")
            return float(Fraction(num, den))
    if "/" in t:
        m = re.fullmatch(r"\s*([-+]?\d+)\s*/\s*([-+]?\d+)\s*", t)
        if not m:
            raise ValueError(f"Bad fraction: {token!r}")
        num, den = int(m.group(1)), int(m.group(2))
        if den == 0:
            raise ValueError("Denominator is zero")
        return float(Fraction(num, den))
    try:
        return float(Decimal(t))
    except InvalidOperation as exc:
        raise ValueError(f"Bad decimal: {token!r}") from exc


def _extract_numeric(text: str) -> float:
    m = _NUMERIC_RE.search((text or "").strip())
    if not m:
        raise ValueError(f"No numeric value in: {text!r}")
    return _parse_numeric(m.group(0))


def _parse_range(gt: str) -> Optional[Tuple[float, float]]:
    if not gt:
        return None
    cleaned = gt.strip().strip("[]").replace("−", "-").replace("–", "-").replace("—", "-")
    pat = _NUMERIC_RE.pattern
    for sep in (r"\s+to\s+", r"\s*-\s*"):
        m = re.fullmatch(rf"\s*({pat}){sep}({pat})\s*", cleaned, re.IGNORECASE)
        if m:
            lo, hi = _parse_numeric(m.group(1)), _parse_numeric(m.group(2))
            return (min(lo, hi), max(lo, hi))
    return None


def _numeric_match(model_answer: str, ground_truth: str, tol: float = 0.02) -> bool:
    if not model_answer:
        return False
    gt = (ground_truth or "").strip().strip("[]")
    try:
        val = _extract_numeric(model_answer)
    except ValueError:
        return False
    bounds = _parse_range(gt)
    if bounds is not None:
        lo, hi = bounds
        return (lo - tol) <= val <= (hi + tol)
    try:
        return abs(val - _extract_numeric(gt)) <= tol
    except ValueError:
        return False


_JUDGE_PROMPT = """\
You are an expert answer evaluator.
Given a Question, Student Answer, and Ground Truth, decide if they are equivalent.
Ignore formatting differences. Return ONLY "YES" or "NO".

QUESTION: {question}
STUDENT ANSWER: {model_answer}
GROUND TRUTH: {ground_truth}
Equivalence (YES/NO):"""


def verify_answer(
    question: str,
    model_answer: str,
    ground_truth: str,
    task: str,
    judge: Optional[ModelConfig] = None,
    numeric_tol: float = 0.02,
) -> bool:
    ma = _normalize_answer_text(model_answer)
    gt = _normalize_answer_text(str(ground_truth or ""))
    qt = canonicalize_task(task)

    # 1) Guard rails
    if not gt or gt.lower() in ("nan", "none", "null", "unknown"):
        return True                     # No ground truth → accept
    if not ma:
        return False

    # 2) MCQ matching (task-based or opportunistic)
    gt_mcq = _extract_mcq_key(gt)
    ma_mcq = _extract_mcq_key(ma)
    if qt in ("mcq_single", "mcq_multiple") or gt_mcq or ma_mcq:
        if not ma_mcq:
            ma_mcq = _resolve_mcq_answer(question, ma, numeric_tol=numeric_tol)
        if not gt_mcq:
            gt_mcq = _resolve_mcq_answer(question, gt, numeric_tol=numeric_tol)
        if ma_mcq and gt_mcq:
            return ma_mcq == gt_mcq

    # 3) Matching columns
    if qt == "matching_columns":
        ma_pairs = _match_pairs(ma)
        gt_pairs = _match_pairs(gt)
        if not ma_pairs or not gt_pairs:
            return False
        return ma_pairs == gt_pairs

    # 4/5) Numeric checks
    if qt in ("numerical_integer", "numerical_range") and _numeric_match(ma, gt, tol=numeric_tol):
        return True

    # Free-form / unknown: try numeric shortcut first
    if qt in ("paragraph", "qa", "unknown"):
        if _numeric_match(ma, gt, tol=numeric_tol):
            return True

    # Catch-all numeric fallback for any task type
    if _numeric_match(ma, gt, tol=numeric_tol):
        return True

    # 6) Normalized string compare
    nm = _normalize_string(ma)
    ng = _normalize_string(gt)
    if nm and ng and nm == ng:
        return True

    # 7) Opportunistic MCQ fallback
    if _is_mcq_like(ma) and _is_mcq_like(gt):
        ma_mcq = _extract_mcq_key(ma)
        gt_mcq = _extract_mcq_key(gt)
        if ma_mcq and gt_mcq:
            return ma_mcq == gt_mcq

    # 8) LLM judge (last resort)
    if judge and judge.url and judge.model:
        prompt = _JUDGE_PROMPT.format(
            question=question, model_answer=ma, ground_truth=gt
        )
        try:
            raw = _llm_complete(
                judge.url, judge.model, prompt,
                api_key=judge.api_key, max_tokens=10, temperature=0.0,
            )
            if re.search(r"^\s*YES\b", raw or "", flags=re.IGNORECASE):
                return True
        except Exception:
            pass

    return False


# ════════════════════════════════════════════════════════════════
#  API CALL  (CoT generation)
# ════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert tutor solving problems across various domains.

Instructions:
1. Think step-by-step through the problem to arrive at the correct solution.
2. Use proper LaTeX notation for all mathematical expressions.
3. End your response with the final answer in the format specified below.

For MCQ (single correct):  [A]
For MCQ (multiple correct): [A,C]
For Numerical (integer):    [42]
For Numerical (decimal):    [3.5]
For Matching:               [1-A, 2-B, 3-C]
For General/Free-form:      [<exact answer>]

For example, if the question is "what is the probability of getting sum 7 when rolling two dice?", a good response would be:
To get a sum of 7, the following pairs are possible: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1). There are 6 favorable outcomes and 36 total outcomes, so the probability is 6/36 = 1/6. Therefore, the final answer is [1/6].

Important:
- Always include the final answer marker at the end.
- For decimal answers round to 2 d.p. but omit trailing zeros (3.5 not 3.50).
- Convert fractions to decimals (3.5 not 7/2)."""


def _call_cot_model(
    cfg: ModelConfig,
    question: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 8000,
    timeout_s: int = 60,
) -> Tuple[str, str, str]:
    """Call the CoT model. Returns (content, reasoning_content, model_answer)."""
    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    session = _get_session()
    resp = session.post(cfg.url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()

    choices = data.get("choices") or []
    if not choices or "message" not in choices[0]:
        raise ValueError("Unexpected API response: missing choices[0].message")

    msg = choices[0]["message"]
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()

    # Wrap reasoning in <think> tags if returned separately
    if reasoning:
        if not reasoning.startswith("<think>"):
            reasoning = f"<think>\n{reasoning}"
        if not reasoning.endswith("</think>"):
            reasoning = f"{reasoning}\n</think>"

    return content, reasoning, extract_model_answer(content)


def _ensure_think_tags(reasoning: str, content: str = "") -> str:
    """Ensure reasoning is returned as a <think>...</think> block."""
    text = (reasoning or "").strip()
    if not text and content:
        match = re.search(r"(?is)<think>\s*(.*?)\s*</think>", content)
        if match:
            text = match.group(1).strip()

    if not text:
        return "<think></think>"

    if "<think>" in text and "</think>" in text:
        return text
    if text.startswith("<think>") and not text.endswith("</think>"):
        return f"{text}\n</think>"
    if text.endswith("</think>") and not text.startswith("<think>"):
        return f"<think>\n{text}"
    return f"<think>\n{text}\n</think>"


def _compose_solution_text(content: str, reasoning: str) -> str:
    """Build a full solution text from model fields.

    Some model responses place the full explanation in reasoning_content and
    keep content as a short final answer token (e.g., "[C]"). In those cases,
    stitch both parts so `solution` always carries full response context.
    """
    c = str(content or "").strip()
    r = str(reasoning or "").strip()

    # If content already looks substantial, keep it as-is.
    if c and (len(c) >= 80 or len(c.split()) >= 16):
        return c

    if r:
        r_block = _ensure_think_tags(r, c)
        if c:
            return f"{r_block}\n\n{c}".strip()
        return r_block
    return c


# ════════════════════════════════════════════════════════════════
#  REJECTION SAMPLING
# ════════════════════════════════════════════════════════════════

def _process_row(
    row: Dict[str, Any],
    model_cfg: ModelConfig,
    judge_cfg: Optional[ModelConfig],
    *,
    temperature: float,
    retry_temperature: float,
    max_tokens: int,
    timeout_s: int,
    max_attempts: int,
) -> QuestionRecord:
    """Process one question with rejection sampling. Thread-safe."""
    row_idx = int(row["_row_idx"])
    question   = str(row["question"])
    ground_truth = str(row.get("ground_truth", ""))
    task = canonicalize_task(row.get("task", "unknown"))

    best: Optional[QuestionRecord] = None
    errors: List[str] = []

    for attempt in range(1, max_attempts + 1):
        temp = temperature if attempt == 1 else retry_temperature
        try:
            content, reasoning, model_answer = _call_cot_model(
                model_cfg, question,
                temperature=temp, max_tokens=max_tokens, timeout_s=timeout_s,
            )
        except Exception as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            continue

        normalized_answer = _normalize_mcq_model_answer(
            question, model_answer, task, ground_truth=ground_truth
        )

        is_correct = verify_answer(question, normalized_answer, ground_truth, task,
                                   judge=judge_cfg)

        record: QuestionRecord = {
            "_row_idx":          row_idx,
            "question":          question,
            "content":           content,
            "reasoning_content": reasoning,
            "model_answer":      normalized_answer,
            "ground_truth":      ground_truth,
            "is_correct":        is_correct,
            "attempts":          attempt,
            "error":             "; ".join(errors),
        }

        if is_correct:
            return record
        best = record

    # All attempts exhausted — return best (or empty shell)
    if best is None:
        best = {
            "_row_idx":          row_idx,
            "question":          question,
            "content":           "",
            "reasoning_content": "",
            "model_answer":      "",
            "ground_truth":      ground_truth,
            "is_correct":        False,
            "attempts":          max_attempts,
            "error":             "; ".join(errors) or "all_attempts_failed",
        }
    else:
        best["error"] = "; ".join(errors)
    return best


def generate_with_rejection_sampling(
    df: pd.DataFrame,
    model_cfg: ModelConfig,
    pipeline_cfg: PipelineConfig,
    judge_cfg: Optional[ModelConfig] = None,
) -> Tuple[List[QuestionRecord], List[QuestionRecord]]:
    """Run rejection sampling over the whole DataFrame.

    Returns (accepted, rejected).
    """
    pending = [row.to_dict() for _, row in df.iterrows()]

    if not pending:
        log.info("No rows to process.")
        return [], []

    log.info("Processing %d questions  workers=%d", len(pending), pipeline_cfg.max_workers)

    accepted: List[QuestionRecord] = []
    rejected: List[QuestionRecord] = []
    done_count = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=pipeline_cfg.max_workers) as ex:

        futs = {
            ex.submit(
                _process_row, row, model_cfg, judge_cfg,
                temperature=pipeline_cfg.temperature,
                retry_temperature=pipeline_cfg.retry_temperature,
                max_tokens=pipeline_cfg.max_tokens,
                timeout_s=pipeline_cfg.timeout_s,
                max_attempts=pipeline_cfg.max_attempts,
            ): row["_row_idx"]
            for row in pending
        }

        for fut in as_completed(futs):
            record = fut.result()
            (accepted if record["is_correct"] else rejected).append(record)

            done_count += 1
            elapsed = time.time() - start
            rate = done_count / elapsed if elapsed > 0 else 0
            eta  = (len(pending) - done_count) / rate if rate > 0 else 0
            status = "OK" if record["is_correct"] else "FAIL"
            log.info(
                "[%s] row=%-5s  ans=%-30s  gold=%-20s  att=%s  "
                "[%d/%d  %.1f q/s  ETA %ds]",
                status,
                record["_row_idx"],
                record.get("model_answer", "")[:30],
                record["ground_truth"][:20],
                record.get("attempts", "?"),
                done_count, len(pending), rate, int(eta),
            )

    return accepted, rejected


# ════════════════════════════════════════════════════════════════
#  STANDARDIZED OUTPUT
# ════════════════════════════════════════════════════════════════

_OUTPUT_COLS = [
    "source", "question", "solution", "<think>",
    "model_answer", "ground_truth", "task", "domain", "language",
    "made by", "model", "SFT/reasoning",
]


def build_output(
    working_df: pd.DataFrame,
    results: List[QuestionRecord],
    output_stem: Path,
    model_name: str,
    made_by: str,
    generation_type: str,
) -> pd.DataFrame:
    if results:
        res_df = pd.DataFrame(results)
        if "_row_idx" in res_df.columns:
            before = len(res_df)
            res_df = res_df.drop_duplicates("_row_idx", keep="last")
            if (dupes := before - len(res_df)):
                log.info("Deduped %d duplicate _row_idx entries", dupes)
    else:
        res_df = pd.DataFrame(columns=["_row_idx","content","reasoning_content",
                                        "model_answer","is_correct"])

    keep = ["_row_idx","content","reasoning_content","model_answer","is_correct"]
    merged = (
        working_df.merge(res_df[keep], on="_row_idx", how="left")
        if "_row_idx" in res_df.columns
        else working_df.copy()
    )

    merged["is_correct"] = merged.get("is_correct", False).fillna(False)
    accepted_only = merged[merged["is_correct"] == True].copy()

    # Keep solution strictly equal to API response `content`.
    accepted_only["solution"] = accepted_only.get("content", "").fillna("")
    accepted_only["<think>"] = [
        _ensure_think_tags(r, c)
        for r, c in zip(
            accepted_only.get("reasoning_content", "").fillna(""),
            accepted_only.get("content", "").fillna(""),
        )
    ]
    accepted_only["made by"] = made_by
    accepted_only["model"] = model_name
    accepted_only["SFT/reasoning"] = generation_type

    for col in _OUTPUT_COLS:
        if col not in accepted_only.columns:
            accepted_only[col] = ""

    final = accepted_only[_OUTPUT_COLS].copy().fillna("")
    parquet_path = output_stem.with_suffix(".parquet")
    final.to_parquet(parquet_path, index=False)
    log.info("Output saved → %s  (accepted rows: %d, total rows: %d)",
             parquet_path, len(final), len(merged))
    return final


# ════════════════════════════════════════════════════════════════
#  PIPELINE RUNNER  (single output per dataset)
# ════════════════════════════════════════════════════════════════

def _dataset_key(input_file: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(input_file).name).replace(".", "_")


def run_dataset(
    working_df: pd.DataFrame,
    model_cfg: ModelConfig,
    pipeline_cfg: PipelineConfig,
    dataset_key: str,
    output_model_name: str,
    judge_cfg: Optional[ModelConfig] = None,
) -> Dict[str, int]:
    pipeline_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    prefix   = f"{dataset_key}_{model_cfg.name}"
    run_dir = pipeline_cfg.output_dir / prefix
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_parquet_path = run_dir / "cot_output.parquet"
    rej_path = run_dir / "rejection_log.json"
    output_stem = run_dir / dataset_key

    log.info("=" * 60)
    log.info("Dataset run: %d rows  → %s", len(working_df), run_dir)

    accepted, rejected = generate_with_rejection_sampling(
        working_df, model_cfg, pipeline_cfg, judge_cfg=judge_cfg
    )

    final_results = accepted + rejected

    if final_results:
        pd.DataFrame(final_results).to_parquet(raw_parquet_path, index=False)
    else:
        pd.DataFrame(columns=["_row_idx", "question", "content", "reasoning_content",
                              "model_answer", "ground_truth", "is_correct",
                              "attempts", "error"]).to_parquet(raw_parquet_path, index=False)

    accepted_total = sum(1 for r in final_results if bool(r.get("is_correct")))
    rejected_rows = sorted(
        int(r["_row_idx"])
        for r in final_results
        if not bool(r.get("is_correct")) and r.get("_row_idx") is not None
    )
    rejected_total = len(rejected_rows)

    with open(rej_path, "w") as fh:
        json.dump({
            "model":    model_cfg.name,
            "dataset":  dataset_key,
            "total":    len(working_df),
            "accepted": accepted_total,
            "rejected": rejected_total,
            "rejected_records": rejected_rows,
        }, fh, indent=2, ensure_ascii=False)

    # Remove legacy names from older layout to keep one consistent file set.
    for legacy_path in (
        run_dir / "cot_output.jsonl",
        run_dir / "enriched.jsonl",
        run_dir / "enriched.parquet",
        run_dir / f"{dataset_key}.jsonl",
    ):
        if legacy_path.exists():
            legacy_path.unlink()

    build_output(
        working_df, final_results,
        output_stem,
        model_name=output_model_name,
        made_by=pipeline_cfg.made_by,
        generation_type=pipeline_cfg.generation_type,
    )

    log.info("Accepted=%d  Rejected=%d  (recent: %d/%d)",
             accepted_total, rejected_total, len(accepted), len(rejected))
    return {"accepted": accepted_total, "rejected": rejected_total}


def get_output_model_label(model_cfg: ModelConfig) -> str:
    if model_cfg.model:
        name = model_cfg.model.split("/")[-1].strip()
        if name:
            return name
    if model_cfg.name:
        return model_cfg.name.replace("_", "-")
    return "gpt-oss-120b"


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def _parse_column_map(args: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in args:
        if "=" not in item:
            raise ValueError(f"Bad column-map format: '{item}' (expected key=value)")
        key, val = item.split("=", 1)
        key = key.strip().lower()
        if key == "answer":
            key = "ground_truth"
        mapping[key] = val.strip()
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generalized CoT Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_file")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--column-map", nargs="+", metavar="KEY=COL")
    parser.add_argument("--no-llm-map",  action="store_true")
    parser.add_argument("--no-enrich",   action="store_true")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--head",        type=int, default=None)
    parser.add_argument("--workers",     type=int, default=None)
    parser.add_argument("--output-dir",  default=None)
    parser.add_argument("--domains",     nargs="+", default=None)
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    # ── Config ──────────────────────────────────────────────────
    config_path = Path(args.config)
    if config_path.exists():
        mapper_cfg, model_cfg, pipeline_cfg = load_config(str(config_path))
        log.info("Config loaded from %s", config_path)
    else:
        log.warning("Config not found at %s, using defaults", config_path)
        mapper_cfg = model_cfg = ModelConfig(url="", model="")
        pipeline_cfg = PipelineConfig()

    if args.output_dir:
        pipeline_cfg.output_dir = Path(args.output_dir)
    if args.workers:
        pipeline_cfg.max_workers = args.workers

    domain_filter: Optional[set[str]] = None
    if args.domains:
        domain_filter = {
            canonicalize_domain_label(d)
            for d in args.domains
            if d.strip()
        }
        domain_filter.discard("unknown")
        if not domain_filter:
            log.error("Domain filter is empty")
            sys.exit(1)

    filter_text = sorted(domain_filter) if domain_filter else "ALL"
    log.info("Workers: %d  |  Domains: %s", pipeline_cfg.max_workers, filter_text)

    # ── Step 1: Load ─────────────────────────────────────────────
    log.info("━" * 60 + "  Step 1: Loading Dataset")
    df = load_dataset(args.input_file)

    # ── Step 2: Column mapping ───────────────────────────────────
    log.info("━" * 60 + "  Step 2: Column Mapping")
    if args.column_map:
        column_map = _parse_column_map(args.column_map)
        log.info("Using explicit column map: %s", column_map)
    else:
        column_map = fuzzy_match_columns(df)
        log.info("Fuzzy-matched: %s", column_map)

        missing = [k for k in REQUIRED_MAPPING if k not in column_map]
        if missing and not args.no_llm_map and mapper_cfg.url and mapper_cfg.model:
            log.info("Missing required after fuzzy %s, trying LLM …", missing)
            try:
                llm_map = map_columns_with_llm(df, mapper_cfg)
                for k, v in llm_map.items():
                    column_map.setdefault(k, v)
                log.info("After LLM merge: %s", column_map)
            except Exception as exc:
                log.warning("LLM mapping failed: %s", exc)

    column_map = confirm_mapping(df, column_map, interactive=args.interactive)

    missing = [k for k in REQUIRED_MAPPING if k not in column_map]
    if missing:
        log.error("Missing required column mappings: %s  (use --column-map)", missing)
        sys.exit(1)

    # ── Step 3: Working DF ───────────────────────────────────────
    log.info("━" * 60 + "  Step 3: Preparing Working DataFrame")
    working_df = prepare_working_df(df, column_map, args.input_file)
    working_df = clean_question_text(working_df)
    if args.head:
        working_df = working_df.head(args.head).copy()
        log.info("Limited to first %d rows", args.head)
    log.info("Working DF: %d rows  cols=%s", len(working_df), list(working_df.columns))

    # ── Step 4: Enrich ───────────────────────────────────────────
    if not args.no_enrich and not args.dry_run and mapper_cfg.url and mapper_cfg.model:
        log.info("━" * 60 + "  Step 4: Metadata Enrichment")
        blank = lambda v: str(v).strip().lower() in ("", "nan", "none", "unknown", "null")  # noqa: E731
        fields = [
            f for f in ("domain", "task", "language")
            if working_df[f].apply(blank).any()
        ]
        if fields:
            working_df = enrich_metadata(working_df, fields, mapper_cfg,
                                         pipeline_cfg.max_workers)
        else:
            log.info("All metadata already populated")
    else:
        log.info("Skipping enrichment")

    # ── Step 5: Domain normalization / optional filter ───────────
    log.info("━" * 60 + "  Step 5: Domain Normalization")
    working_df["domain"] = working_df["domain"].apply(canonicalize_domain_label)
    dist = working_df["domain"].value_counts(dropna=False).to_dict()
    log.info("Domain distribution: %s", dist)

    if domain_filter:
        before = len(working_df)
        working_df = working_df[working_df["domain"].isin(domain_filter)].copy()
        dropped = before - len(working_df)
        if dropped:
            log.info("Dropped %d rows outside target domains", dropped)
    else:
        log.info("No domain filter provided; keeping all rows from input file")

    log.info("Rows remaining: %d", len(working_df))
    if working_df.empty:
        log.error("No rows left after filtering")
        sys.exit(1)

    # ── Dry run ──────────────────────────────────────────────────
    if args.dry_run:
        log.info("━" * 60 + "  DRY RUN — skipping API calls")
        key = _dataset_key(args.input_file)
        prefix = f"{key}_{model_cfg.name or 'model'}"
        run_dir = pipeline_cfg.output_dir / prefix
        log.info("  Rows to process: %d", len(working_df))
        log.info("  Output folder: %s", run_dir)
        log.info("  Raw CoT Parquet: %s", run_dir / "cot_output.parquet")
        log.info("  Final Parquet: %s", run_dir / f"{key}.parquet")
        log.info("  Rejection Log: %s", run_dir / "rejection_log.json")
        log.info("Preview:\n%s", working_df.head(5).to_string(index=False))
        log.info("Dry run complete.")
        return

    # ── Step 6: Run ──────────────────────────────────────────────
    log.info("━" * 60 + "  Step 6: Running CoT Pipeline")
    if not model_cfg.url or not model_cfg.model:
        log.error("No model configured. Set 'model' section in config.yaml")
        sys.exit(1)

    key = _dataset_key(args.input_file)
    pipeline_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Model: %s (%s)  dataset: %s  workers: %d",
             model_cfg.name, model_cfg.model, key, pipeline_cfg.max_workers)

    output_model_name = get_output_model_label(model_cfg)

    t0 = time.time()
    summary = run_dataset(
        working_df, model_cfg, pipeline_cfg, key, output_model_name,
        judge_cfg=mapper_cfg if mapper_cfg.url else None,
    )
    total_accepted = summary["accepted"]
    total_rejected = summary["rejected"]

    elapsed = time.time() - t0
    processed = len(working_df)
    prefix = f"{key}_{model_cfg.name}"
    run_dir = pipeline_cfg.output_dir / prefix

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE  model=%s", model_cfg.name)
    log.info("Output folder: %s", run_dir)
    log.info("Output files:")
    log.info("  %s", run_dir / "cot_output.parquet")
    log.info("  %s", run_dir / f"{key}.parquet")
    log.info("  %s", run_dir / "rejection_log.json")
    log.info("Total: %d accepted  %d rejected  in %.1fs  (%.1f q/s)",
             total_accepted, total_rejected, elapsed,
             processed / elapsed if elapsed else 0)


if __name__ == "__main__":
    main()