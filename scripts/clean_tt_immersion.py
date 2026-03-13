"""Clean Tommee Tippee immersion JSON into analysis-ready CSV.

Source : data/raw/dataset_tommee_tippee.json
Output : data/processed/tt_immersion_clean.csv

Fields added on top of baseline clean_data.py output:
  title_raw, brand_raw, price_value, price_currency, price_flag,
  product_type, product_family,
  is_tommee_tippee_brand, is_competitor_brand,
  is_compatible_accessory, is_core_bottle_listing,
  search_term (placeholder — dataset has no keyword field)
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "dataset_tommee_tippee.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_CSV = PROCESSED_DIR / "tt_immersion_clean.csv"

CHUNK_SIZE = 1024 * 256

# ---------------------------------------------------------------------------
# Brand normalisation (mirrors app.py BRAND_NORMALIZATION_MAP)
# ---------------------------------------------------------------------------
BRAND_NORMALIZATION_MAP: Dict[str, str] = {
    "tommee tippee": "Tommee Tippee",
    "tommee tipee": "Tommee Tippee",
    "tommee-tippee": "Tommee Tippee",
    "dr. brown's": "Dr. Brown's",
    "dr browns": "Dr. Brown's",
    "dr. browns": "Dr. Brown's",
    "dr brown's": "Dr. Brown's",
    "dr.brown's": "Dr. Brown's",
    "philips avent": "Philips Avent",
    "avent": "Philips Avent",
    "evenflo feeding": "Evenflo",
    "the first years": "The First Years",
    "first years": "The First Years",
    "nanobébé": "Nanobebe",
    "nuk usa": "NUK",
}

TOMMEE_TIPPEE_VARIANTS = re.compile(
    r"tommee\s*ti(?:pp?|p)ee", re.IGNORECASE
)


def normalize_brand(raw: object) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "Unknown"
    lower = raw.strip().lower()
    for key, val in BRAND_NORMALIZATION_MAP.items():
        if lower == key:
            return val
    return raw.strip().title()


# ---------------------------------------------------------------------------
# Product-type classification
# ---------------------------------------------------------------------------
PRODUCT_TYPE_RULES: List[Tuple[str, str]] = [
    # order matters — first match wins
    ("bundle",          r"\bstarter\s+kit\b|\bvalue\s+set\b|\bcomplete\s+kit\b|\bgift\s+set\b"),
    ("bottle_set",      r"\b(?:\d+[-\s]?pack|pack\s+of\s+\d+|set\s+of\s+\d+)\b.*\bbottle\b|\bbottle\b.*\b(?:\d+[-\s]?pack|pack\s+of\s+\d+|set\s+of\s+\d+)\b"),
    ("baby_bottle",     r"\bbabies?\s+bottle\b|\bbaby\s+bottle\b|\bfeeding\s+bottle\b|\banti[-\s]?colic\s+bottle\b|\bbottle\b"),
    ("nipple",          r"\bnipple\b|\bteat\b"),
    ("pacifier",        r"\bpacifier\b|\bdummy\b|\bsoother\b"),
    ("bottle_warmer",   r"\bwarmer\b|\bwarming\b"),
    ("sterilizer",      r"\bsteril[iz]\b|\bsteriliser\b|\bsterilizer\b"),
    ("cup",             r"\bsippy\s+cup\b|\btraining\s+cup\b|\blearner\s+cup\b|\bstraw\s+cup\b|\btransition\s+cup\b"),
    ("breast_pump",     r"\bbreast\s+pump\b|\belectric\s+pump\b"),
    ("accessory",       r"\bbrush\b|\bbottle\s+brush\b|\binsert\b|\badapter\b|\bclip\b|\bstrap\b|\bcover\b|\bcap\b"),
    ("replacement_part",r"\breplacement\b|\bspare\b"),
]

COMPILED_PRODUCT_TYPE_RULES = [
    (ptype, re.compile(pattern, re.IGNORECASE))
    for ptype, pattern in PRODUCT_TYPE_RULES
]


def classify_product_type(title: str, breadcrumbs: str = "") -> str:
    text = f"{title} {breadcrumbs}".lower()
    for ptype, pattern in COMPILED_PRODUCT_TYPE_RULES:
        if pattern.search(text):
            return ptype
    return "unknown"


# ---------------------------------------------------------------------------
# Product family derivation (strip variant noise from title)
# ---------------------------------------------------------------------------
_VARIANT_NOISE = re.compile(
    r"""
    \b\d+\s*(?:oz|ml|fl\.?\s*oz)\b   # volume
    | \b(?:\d+[-\s]?pack|pack\s+of\s+\d+|set\s+of\s+\d+|\d+\s+(?:bottles?|nipples?|teats?))\b  # quantity
    | \b(?:pink|blue|green|red|purple|white|grey|gray|clear|black|yellow|orange|teal|navy)\b    # colour
    | \bsize\s+\d+\b                  # size numbers
    | ,\s*\d+\s*(?:oz|ml)\b           # ", 4 oz"
    """,
    re.IGNORECASE | re.VERBOSE,
)


def derive_product_family(title: str) -> str:
    if not isinstance(title, str):
        return ""
    family = _VARIANT_NOISE.sub(" ", title)
    family = re.sub(r"\s{2,}", " ", family).strip(" ,;-")
    return family


# ---------------------------------------------------------------------------
# Brand/flag classification
# ---------------------------------------------------------------------------
def flag_tommee_tippee(brand: str, title: str) -> bool:
    if TOMMEE_TIPPEE_VARIANTS.search(brand):
        return True
    if TOMMEE_TIPPEE_VARIANTS.search(title):
        return True
    return False


_COMPETITOR_BRANDS = re.compile(
    r"\b(?:dr\.?\s*brown'?s?|philips\s+avent|avent|evenflo|nuk|mam|comotomo|"
    r"nanobebe|nanob[eé]b[eé]|lansinoh|medela|chicco|playtex|gerber|"
    r"the\s+first\s+years?|first\s+years?)\b",
    re.IGNORECASE,
)


def flag_competitor(brand: str, title: str) -> bool:
    text = f"{brand} {title}"
    return bool(_COMPETITOR_BRANDS.search(text))


def flag_compatible_accessory(title: str, breadcrumbs: str = "") -> bool:
    text = f"{title} {breadcrumbs}".lower()
    return bool(re.search(r"\b(?:compatible\s+with|fits\s+(?:most|all)?|works\s+with|for\s+tommee\s+tippee)\b", text, re.IGNORECASE))


def flag_core_bottle(product_type: str) -> bool:
    return product_type in {"baby_bottle", "bottle_set"}


# ---------------------------------------------------------------------------
# Price flag
# ---------------------------------------------------------------------------
def price_flag(price_value: float) -> str:
    if np.isnan(price_value):
        return "unknown"
    if price_value < 10:
        return "budget"
    if price_value < 25:
        return "mid"
    if price_value < 50:
        return "premium"
    return "luxury"


# ---------------------------------------------------------------------------
# JSON streaming helpers (reused pattern from clean_data.py)
# ---------------------------------------------------------------------------
def iter_json_array(path: Path) -> Generator[dict, None, None]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        buf = f.read(CHUNK_SIZE)
        idx = 0

        # skip to opening bracket
        while idx < len(buf) and buf[idx].isspace():
            idx += 1
        if not buf or buf[idx] != "[":
            raise ValueError("Expected JSON array starting with '['")
        idx += 1

        while True:
            # ensure buffer has content
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


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------
def to_numeric(value: object) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return np.nan
        m = re.search(r"-?\d+(\.\d+)?", text)
        return float(m.group(0)) if m else np.nan
    return np.nan


def clean_text(value: object) -> object:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    return value


def split_breadcrumbs(value: object) -> Tuple[object, object]:
    if not isinstance(value, str) or not value.strip():
        return np.nan, np.nan
    parts = [p.strip() for p in value.split(">") if p.strip()]
    return (parts[0], parts[-1]) if parts else (np.nan, np.nan)


def build_row(record: dict) -> Dict[str, object]:
    row: Dict[str, object] = {}

    # ---- raw fields -------------------------------------------------------
    title_raw = clean_text(record.get("title", ""))
    row["title_raw"] = title_raw
    row["title"] = title_raw  # primary title column

    row["asin"] = clean_text(record.get("asin", ""))
    row["url"] = clean_text(record.get("url", ""))

    brand_raw = clean_text(record.get("brand", ""))
    row["brand_raw"] = brand_raw
    row["brand"] = normalize_brand(brand_raw)

    row["stars"] = to_numeric(record.get("stars"))
    if not np.isnan(row["stars"]) and not (0 <= row["stars"] <= 5):
        row["stars"] = np.nan

    row["reviewsCount"] = to_numeric(record.get("reviewsCount"))
    if not np.isnan(row["reviewsCount"]) and row["reviewsCount"] < 0:
        row["reviewsCount"] = np.nan

    # ---- price (nested object OR flat) ------------------------------------
    price_obj = record.get("price")
    if isinstance(price_obj, dict):
        row["price_value"] = to_numeric(price_obj.get("value"))
        currency_raw = price_obj.get("currency", "")
        row["price_currency"] = str(currency_raw).strip() if currency_raw else "USD"
    elif isinstance(price_obj, (int, float)):
        row["price_value"] = to_numeric(price_obj)
        row["price_currency"] = "USD"
    elif isinstance(price_obj, str):
        row["price_value"] = to_numeric(price_obj)
        row["price_currency"] = "USD"
    else:
        row["price_value"] = np.nan
        row["price_currency"] = "USD"

    if not np.isnan(row["price_value"]) and row["price_value"] <= 0:
        row["price_value"] = np.nan

    row["price_flag"] = price_flag(row["price_value"])

    # ---- text fields ------------------------------------------------------
    breadcrumbs = clean_text(record.get("breadCrumbs") or record.get("breadcrumbs", ""))
    row["breadcrumbs"] = breadcrumbs

    description = clean_text(record.get("description", ""))
    row["description"] = description
    row["description_length"] = len(description) if isinstance(description, str) else 0

    row["thumbnailImage"] = clean_text(record.get("thumbnailImage", ""))

    # ---- derived fields ---------------------------------------------------
    title_str = title_raw if isinstance(title_raw, str) else ""
    breadcrumbs_str = breadcrumbs if isinstance(breadcrumbs, str) else ""

    row["clean_title"] = title_str.lower().strip()
    row["title_word_count"] = len(title_str.split()) if title_str else 0

    cat_root, cat_leaf = split_breadcrumbs(breadcrumbs_str)
    row["category_root"] = cat_root
    row["category_leaf"] = cat_leaf

    reviews_for_score = row["reviewsCount"] if not np.isnan(row["reviewsCount"]) else 0.0
    stars_for_score = row["stars"] if not np.isnan(row["stars"]) else 0.0
    row["popularity_score"] = stars_for_score * math.log1p(reviews_for_score)

    product_type = classify_product_type(title_str, breadcrumbs_str)
    row["product_type"] = product_type

    row["product_family"] = derive_product_family(title_str)

    brand_str = row["brand"] if isinstance(row["brand"], str) else ""
    row["is_tommee_tippee_brand"] = flag_tommee_tippee(brand_str, title_str)
    row["is_competitor_brand"] = (
        flag_competitor(brand_str, title_str) and not row["is_tommee_tippee_brand"]
    )
    row["is_compatible_accessory"] = flag_compatible_accessory(title_str, breadcrumbs_str)
    row["is_core_bottle_listing"] = flag_core_bottle(product_type)

    # placeholder — source dataset has no search_term/keyword field
    row["search_term"] = np.nan

    return row


# ---------------------------------------------------------------------------
# Deduplication: ASIN → URL → normalised title
# ---------------------------------------------------------------------------
def dedupe_key(row: Dict[str, object]) -> str:
    asin = row.get("asin", "")
    if isinstance(asin, str) and asin.strip():
        return f"asin:{asin.strip().upper()}"
    url = row.get("url", "")
    if isinstance(url, str) and url.strip():
        return f"url:{url.strip()}"
    title = row.get("clean_title", "")
    if isinstance(title, str) and title.strip():
        return f"title:{re.sub(r'[^a-z0-9]', '', title.strip())}"
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
COLUMN_ORDER = [
    "asin", "url", "title", "title_raw", "clean_title", "title_word_count",
    "product_family", "brand", "brand_raw",
    "stars", "reviewsCount", "price_value", "price_currency", "price_flag",
    "category_root", "category_leaf", "breadcrumbs",
    "description", "description_length", "popularity_score",
    "product_type",
    "is_tommee_tippee_brand", "is_competitor_brand",
    "is_compatible_accessory", "is_core_bottle_listing",
    "thumbnailImage", "search_term",
]


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_FILE}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    seen: set = set()

    for record in iter_json_array(RAW_FILE):
        row = build_row(record)
        key = dedupe_key(row)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        rows.append(row)

    if not rows:
        raise RuntimeError("No records parsed from dataset.")

    df = pd.DataFrame(rows)

    existing_cols = [c for c in COLUMN_ORDER if c in df.columns]
    extra_cols = [c for c in df.columns if c not in COLUMN_ORDER]
    df = df[existing_cols + extra_cols]

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    n_tt = df["is_tommee_tippee_brand"].sum()
    n_comp = df["is_competitor_brand"].sum()
    n_bottle = df["is_core_bottle_listing"].sum()
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Rows: {len(df):,}  |  Columns: {len(df.columns)}")
    print(f"  Tommee Tippee listings : {n_tt}")
    print(f"  Competitor listings    : {n_comp}")
    print(f"  Core bottle listings   : {n_bottle}")
    pt_counts = df["product_type"].value_counts()
    print("\nProduct type breakdown:")
    for pt, cnt in pt_counts.items():
        print(f"  {pt:<25} {cnt}")


if __name__ == "__main__":
    main()
