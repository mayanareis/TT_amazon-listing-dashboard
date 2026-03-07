"""Clean Amazon raw JSON into analysis-ready CSV."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
OUTPUT_CSV = CLEANED_DIR / "amazon_products_clean.csv"
CHUNK_SIZE = 1024 * 256


def detect_single_json_file(raw_dir: Path) -> Path:
    files = sorted(raw_dir.glob("*.json"))
    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one JSON file in {raw_dir}. Found {len(files)}."
        )
    return files[0]


def first_non_whitespace_char(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        while True:
            char = f.read(1)
            if not char:
                return ""
            if not char.isspace():
                return char


def detect_structure(path: Path) -> Tuple[str, Optional[str]]:
    first_char = first_non_whitespace_char(path)
    if first_char == "[":
        return "json_array", None
    if first_char == "{":
        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            header = f.read(CHUNK_SIZE)
        if re.search(r'"items"\s*:\s*\[', header):
            return "wrapped_object_array", "items"
        if re.search(r'"data"\s*:\s*\[', header):
            return "wrapped_object_array", "data"
        return "json_object", None
    return "jsonl", None


def find_array_start(path: Path, key: Optional[str]) -> int:
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        text = f.read(CHUNK_SIZE)
        if key is None:
            pos = text.find("[")
            if pos == -1:
                raise ValueError("Could not find '[' in JSON file.")
            return pos
        match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', text)
        if not match:
            raise ValueError(f'Could not find wrapped key "{key}" with array.')
        return match.end() - 1


def iter_json_array_values(path: Path, array_start: int) -> Generator[dict, None, None]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        f.seek(array_start)
        buf = f.read(CHUNK_SIZE)
        if not buf:
            return

        idx = 0
        while True:
            while idx < len(buf) and buf[idx].isspace():
                idx += 1
            if idx < len(buf):
                break
            more = f.read(CHUNK_SIZE)
            if not more:
                return
            buf += more

        if buf[idx] != "[":
            raise ValueError("Expected '[' at array start.")
        idx += 1

        while True:
            while True:
                while idx < len(buf) and buf[idx].isspace():
                    idx += 1
                if idx < len(buf):
                    break
                more = f.read(CHUNK_SIZE)
                if not more:
                    return
                buf += more

            if buf[idx] == "]":
                return
            if buf[idx] == ",":
                idx += 1
                continue

            while True:
                try:
                    value, end_idx = decoder.raw_decode(buf, idx)
                    idx = end_idx
                    if isinstance(value, dict):
                        yield value
                    break
                except json.JSONDecodeError:
                    more = f.read(CHUNK_SIZE)
                    if not more:
                        return
                    buf += more

            if idx > CHUNK_SIZE:
                buf = buf[idx:]
                idx = 0


def iter_jsonl(path: Path) -> Generator[dict, None, None]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def iter_records(path: Path) -> Generator[dict, None, None]:
    structure, key = detect_structure(path)
    if structure == "jsonl":
        yield from iter_jsonl(path)
        return

    if structure in {"json_array", "wrapped_object_array"}:
        array_start = find_array_start(path, key)
        yield from iter_json_array_values(path, array_start)
        return

    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        yield obj


def flatten_dict(data: dict, parent: str = "", sep: str = "/") -> Dict[str, object]:
    flat: Dict[str, object] = {}
    for key, value in data.items():
        key_str = str(key)
        new_key = f"{parent}{sep}{key_str}" if parent else key_str
        if isinstance(value, dict):
            flat.update(flatten_dict(value, parent=new_key, sep=sep))
        else:
            flat[new_key] = value
    return flat


def standardize_name(name: str) -> str:
    clean = name.strip().lower()
    clean = re.sub(r"[ /.-]+", "_", clean)
    clean = re.sub(r"[^a-z0-9_]", "", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean


def to_numeric(value: object) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if text == "":
            return np.nan
        match = re.search(r"-?\d+(\.\d+)?", text)
        if not match:
            return np.nan
        try:
            return float(match.group(0))
        except ValueError:
            return np.nan
    return np.nan


def clean_text(value: object) -> object:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    return value


def split_breadcrumbs(value: object) -> Tuple[object, object]:
    if not isinstance(value, str) or not value.strip():
        return np.nan, np.nan
    parts = [part.strip() for part in value.split(">") if part.strip()]
    if not parts:
        return np.nan, np.nan
    return parts[0], parts[-1]


def build_clean_row(record: dict) -> Dict[str, object]:
    flat = flatten_dict(record)
    standardized = {standardize_name(k): v for k, v in flat.items()}

    row: Dict[str, object] = {}
    source_map = {
        "title": "title",
        "asin": "asin",
        "brand": "brand",
        "stars": "stars",
        "reviewsCount": "reviewscount",
        "thumbnailimage": "thumbnailimage",
        "breadcrumbs": "breadcrumbs",
        "description": "description",
        "price": "price",
        "price_currency": "price_currency",
        "price_value": "price_value",
        "url": "url",
    }
    for out_col, src_col in source_map.items():
        col = src_col
        if col in standardized:
            row[out_col] = standardized[col]

    # Normalize text values.
    for col in ["title", "asin", "brand", "thumbnailimage", "breadcrumbs", "description", "url"]:
        row[col] = clean_text(row.get(col))

    # Numeric conversion.
    row["stars"] = to_numeric(row.get("stars"))
    row["reviewsCount"] = to_numeric(row.get("reviewsCount"))
    row["price_value"] = to_numeric(row.get("price_value"))

    # Rule-based value validation.
    if pd.notna(row["stars"]) and not (0 <= row["stars"] <= 5):
        row["stars"] = np.nan
    if pd.notna(row["reviewsCount"]) and row["reviewsCount"] < 0:
        row["reviewsCount"] = np.nan
    if pd.notna(row["price_value"]) and row["price_value"] <= 0:
        row["price_value"] = np.nan

    # Derived fields.
    title = row.get("title")
    row["clean_title"] = clean_text(title.lower()) if isinstance(title, str) else np.nan
    row["title_word_count"] = len(title.split()) if isinstance(title, str) and title else 0

    cat_root, cat_leaf = split_breadcrumbs(row.get("breadcrumbs"))
    row["category_root"] = cat_root
    row["category_leaf"] = cat_leaf

    reviews_for_score = row["reviewsCount"] if pd.notna(row["reviewsCount"]) else 0.0
    stars_for_score = row["stars"] if pd.notna(row["stars"]) else 0.0
    row["popularity_score"] = stars_for_score * math.log1p(reviews_for_score)

    description = row.get("description")
    row["description_length"] = len(description) if isinstance(description, str) else 0

    if not isinstance(row.get("brand"), str) or row["brand"] == "":
        row["brand"] = "Unknown"

    return row


def main() -> None:
    raw_file = detect_single_json_file(RAW_DIR)
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_rows: List[Dict[str, object]] = []
    seen_asin = set()

    for record in iter_records(raw_file):
        row = build_clean_row(record)

        asin = row.get("asin")
        if isinstance(asin, str) and asin:
            if asin in seen_asin:
                continue
            seen_asin.add(asin)

        cleaned_rows.append(row)

    if not cleaned_rows:
        raise RuntimeError("No rows were parsed from the raw dataset.")

    df = pd.DataFrame(cleaned_rows)

    # Keep only columns that exist plus derived features.
    desired_order = [
        "title",
        "clean_title",
        "title_word_count",
        "asin",
        "brand",
        "stars",
        "reviewsCount",
        "price_value",
        "price_currency",
        "price",
        "category_root",
        "category_leaf",
        "breadcrumbs",
        "description",
        "description_length",
        "popularity_score",
        "thumbnailimage",
        "url",
    ]
    existing_cols = [col for col in desired_order if col in df.columns]
    df = df[existing_cols]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved cleaned data to: {OUTPUT_CSV}")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
