"""Lightweight raw JSON inspection script for Amazon dataset."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
SAMPLE_ROWS = 3
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

    # Fallback: likely JSONL
    return "jsonl", None


def find_array_start(path: Path, key: Optional[str]) -> int:
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        text = f.read(CHUNK_SIZE)
        if key is None:
            bracket = text.find("[")
            if bracket == -1:
                raise ValueError("Could not find array start '[' in file.")
            return bracket

        key_match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', text)
        if not key_match:
            raise ValueError(f'Could not find wrapped key "{key}" with array value.')
        return key_match.end() - 1


def iter_json_array_values(path: Path, array_start: int) -> Generator[dict, None, None]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        f.seek(array_start)
        buf = f.read(CHUNK_SIZE)
        if not buf:
            return

        idx = 0
        # Move to first '['
        while True:
            while idx < len(buf) and buf[idx].isspace():
                idx += 1
            if idx < len(buf):
                break
            more = f.read(CHUNK_SIZE)
            if not more:
                return
            buf += more

        if idx >= len(buf) or buf[idx] != "[":
            raise ValueError("Expected '[' at start of array.")
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


def sample_records(path: Path, structure: str, key: Optional[str], n: int) -> List[dict]:
    rows: List[dict] = []
    if structure == "jsonl":
        source = iter_jsonl(path)
    elif structure in {"json_array", "wrapped_object_array"}:
        array_start = find_array_start(path, key)
        source = iter_json_array_values(path, array_start)
    else:
        # Tiny fallback for non-standard object files.
        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            obj = json.load(f)
        source = []
        if isinstance(obj, dict):
            source = [obj]

    for record in source:
        rows.append(record)
        if len(rows) >= n:
            break
    return rows


def approximate_row_count(path: Path, structure: str, key: Optional[str]) -> Optional[int]:
    try:
        if structure == "jsonl":
            count = 0
            with path.open("r", encoding="utf-8-sig", errors="replace") as f:
                for line in f:
                    if line.strip():
                        count += 1
            return count

        if structure in {"json_array", "wrapped_object_array"}:
            array_start = find_array_start(path, key)
            return sum(1 for _ in iter_json_array_values(path, array_start))
    except Exception:
        return None
    return None


def main() -> None:
    raw_file = detect_single_json_file(RAW_DIR)
    size_mb = raw_file.stat().st_size / (1024 * 1024)
    print(f"Raw file: {raw_file}")
    print(f"File size: {size_mb:.2f} MB")

    structure, key = detect_structure(raw_file)
    detail = f" ({key})" if key else ""
    print(f"Detected structure: {structure}{detail}")

    records = sample_records(raw_file, structure, key, SAMPLE_ROWS)
    if not records:
        print("No sample records could be parsed.")
        return

    sample_df = pd.json_normalize(records, sep="/")
    print("\nFirst 3 rows (sample):")
    print(sample_df.head(3).to_string(index=False))

    print("\nColumn names (sample):")
    print(sample_df.columns.tolist())

    approx_count = approximate_row_count(raw_file, structure, key)
    if approx_count is None:
        print("\nApproximate row count: could not estimate safely.")
    else:
        print(f"\nApproximate row count: {approx_count}")

    print("\nMissing values in sample:")
    print(sample_df.isna().sum().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
