"""Tommee Tippee — Amazon Competitive Intelligence Dashboard (Client View).

A curated, client-facing dashboard focused on Tommee Tippee's position in the
Amazon baby bottle market. Cleaner layout, fewer tabs, TT-first framing.

Tabs:
  Market Overview | Title Intelligence | Messaging & Claims |
  Branded Search  | Product Explorer

Original internal dashboard: app.py (untouched).
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tommee Tippee — Amazon Competitive Intelligence",
    layout="wide",
    page_icon="🍼",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_FILE = PROJECT_ROOT / "data" / "cleaned" / "amazon_products_clean.csv"
TT_IMMERSION_FILE = PROJECT_ROOT / "data" / "processed" / "tt_immersion_clean.csv"

# Brand colour constants used throughout all charts
TT_BLUE = "#0066cc"
COMP_ORANGE = "#e05c2a"


# ══════════════════════════════════════════════════════════════════════════════
# PHRASE MAPS
# Order matters — longer / more specific phrases first so they are matched
# before shorter component words can be consumed separately.
# ══════════════════════════════════════════════════════════════════════════════

TITLE_PHRASE_MAP: List[Tuple[str, str]] = [
    (r"\btommee\s+tippee\b",         "tommee_tippee"),
    (r"\bdr\.?\s+brown'?s?\b",       "dr_browns"),
    (r"\bdr\.?\s+brown\b",           "dr_brown"),
    (r"\bairfree\s+vent\b",          "airfree_vent"),
    (r"\bairfree\b",                 "airfree_vent"),
    (r"\bair[-\s]+free\b",           "air_free"),
    (r"\bnatural\s+response\b",      "natural_response"),
    (r"\bnatural\s+bottle\b",        "natural_bottle"),
    (r"\bnatural\s+latch\b",         "natural_latch"),
    (r"\bglass\s+bottle\b",          "glass_bottle"),
    (r"\bbaby\s+bottles?\b",         "baby_bottle"),
    (r"\bsilicone\s+nipple\b",       "silicone_nipple"),
    (r"\banti[-\s]+colic\b",         "anti_colic"),
    (r"\bbpa[-\s]+free\b",           "bpa_free"),
    (r"\bleak[-\s]+proof\b",         "leak_proof"),
    (r"\bdishwasher[-\s]+safe\b",    "dishwasher_safe"),
    (r"\bbreast[-\s]+like\b",        "breast_like"),
    (r"\beasy\s+hold\b",             "easy_hold"),
    (r"\beasy\s+grip\b",             "easy_grip"),
    (r"\bslow\s+flow\b",             "slow_flow"),
    (r"\bmedium\s+flow\b",           "medium_flow"),
    (r"\bfast\s+flow\b",             "fast_flow"),
    (r"\bfirst\s+flow\b",            "first_flow"),
    (r"\bwide\s+neck\b",             "wide_neck"),
    (r"\bbreast\s+to\s+bottle\b",    "breast_to_bottle"),
    (r"\bbottle\s+to\s+breast\b",    "breast_to_bottle"),
    (r"\bself[-\s]+steriliz\w*\b",   "self_sterilizing"),
    (r"\bseamless\s+transition\b",   "seamless_transition"),
    (r"\bsippy\s+cup\b",             "sippy_cup"),
    (r"\bsip\s+cup\b",               "sippy_cup"),
]

DESC_PHRASE_MAP: List[Tuple[str, str]] = [
    (r"\btommee\s+tippee\b",         "tommee_tippee"),
    (r"\bdr\.?\s+brown'?s?\b",       "dr_browns"),
    (r"\bdr\.?\s+brown\b",           "dr_brown"),
    (r"\bairfree\s+vent\b",          "airfree_vent"),
    (r"\bairfree\b",                 "airfree_vent"),
    (r"\bair[-\s]+free\b",           "air_free"),
    (r"\bnatural\s+response\b",      "natural_response"),
    (r"\bnatural\s+latch\b",         "natural_latch"),
    (r"\bbottle[-\s]+feeding\b",     "bottle_feeding"),
    (r"\bfeeding\s+bottle\b",        "feeding_bottle"),
    (r"\bbreast\s*feeding\b",        "breastfeeding"),
    (r"\bbreast[-\s]+like\b",        "breast_like"),
    (r"\bbreast\s+feed\b",           "breastfeed"),
    (r"\bglass\s+bottle\b",          "glass_bottle"),
    (r"\bbaby\s+bottles?\b",         "baby_bottle"),
    (r"\bsilicone\s+nipple\b",       "silicone_nipple"),
    (r"\bsoft\s+silicone\b",         "soft_silicone"),
    (r"\banti[-\s]+colic\b",         "anti_colic"),
    (r"\bbpa[-\s]+free\b",           "bpa_free"),
    (r"\bleak[-\s]+proof\b",         "leak_proof"),
    (r"\bdishwasher[-\s]+safe\b",    "dishwasher_safe"),
    (r"\beasy[-\s]+to[-\s]+clean\b", "easy_clean"),
    (r"\beasy[-\s]+clean\b",         "easy_clean"),
    (r"\beasy\s+hold\b",             "easy_hold"),
    (r"\beasy\s+grip\b",             "easy_grip"),
    (r"\bslow\s+flow\b",             "slow_flow"),
    (r"\bmedium\s+flow\b",           "medium_flow"),
    (r"\bfast\s+flow\b",             "fast_flow"),
    (r"\bfirst\s+flow\b",            "first_flow"),
    (r"\bwide\s+neck\b",             "wide_neck"),
    (r"\bupright\s+feeding\b",       "upright_feeding"),
    (r"\bbreast\s+to\s+bottle\b",    "breast_to_bottle"),
    (r"\bbottle\s+to\s+breast\b",    "breast_to_bottle"),
    (r"\bself[-\s]+steriliz\w*\b",   "self_sterilizing"),
    (r"\bseamless\s+transition\b",   "seamless_transition"),
    (r"\bvent\s+system\b",           "vent_system"),
    (r"\banti[-\s]?colic\s+system\b", "anti_colic_system"),
    (r"\bsoft\s+nipple\b",           "soft_nipple"),
    (r"\bnatural\s+nipple\b",        "natural_nipple"),
    (r"\bsippy\s+cup\b",             "sippy_cup"),
    (r"\bsip\s+cup\b",               "sippy_cup"),
]

TITLE_STOPWORDS: set = {
    "the", "and", "for", "with", "this", "that", "from", "your", "you",
    "are", "was", "were", "have", "has", "had", "will", "can", "our",
    "its", "all", "not", "but", "use", "using", "new", "amazon",
    "in", "on", "of", "to", "a", "an", "is", "it", "be",
    "or", "by", "at", "as", "up", "do", "if", "so", "no", "we", "my",
    "he", "she", "they", "them", "their", "also", "into", "than", "other",
    "each", "more", "made", "make", "when", "which", "who", "what", "how",
    "get", "give", "just", "out", "any", "one", "two", "may", "well",
    "baby", "bottle", "bottles",
    "oz", "ml", "pk", "count",
    "month", "months",
    "pack", "set", "piece", "pieces", "product",
    "tommee", "tippee", "brown", "dr",
    "philips", "avent", "nuk", "evenflo", "mam",
    "comotomo", "nanobebe", "lansinoh", "medela",
    "chicco", "playtex", "gerber", "first", "years",
}

DESC_STOPWORDS: set = {
    "the", "and", "for", "with", "that", "this", "your", "from", "into",
    "only", "are", "has", "have", "had", "will", "can", "our", "its",
    "all", "not", "but", "use", "used", "using", "new", "amazon",
    "in", "on", "of", "to", "a", "an", "is", "it", "be", "or", "by",
    "at", "as", "up", "do", "if", "so", "no", "we", "my", "you", "was",
    "were", "also", "each", "more", "when", "which", "who", "what", "how",
    "get", "just", "any", "one", "two", "may", "well", "than", "other",
    "made", "make", "them", "they", "their", "while", "both",
    "designed", "includes", "included", "featuring", "features", "help",
    "helps", "helping", "keep", "keeps", "allow", "allows", "give", "gives",
    "reduce", "reduces", "providing", "provides", "ensure", "ensures",
    "prevent", "prevents", "support", "supports", "comes", "come",
    "baby", "bottle", "bottles", "feeding", "product", "products",
    "nipple", "nipples",
    "oz", "ml",
    "tommee", "tippee", "brown", "dr",
    "philips", "avent", "nuk", "evenflo", "mam",
    "comotomo", "nanobebe", "lansinoh", "medela",
    "chicco", "playtex", "gerber", "first", "years",
}

CLAIM_MAP: dict = {
    "Anti-Colic":       r"\banti[-\s]?colic\b",
    "BPA Free":         r"\bbpa[-\s]?free\b",
    "Natural Response": r"\bnatural\s+response\b",
    "Leak Proof":       r"\bleak[-\s]?proof\b",
    "Dishwasher Safe":  r"\bdishwasher[-\s]?safe\b",
    "Easy Clean":       r"\beasy[-\s]?(?:to[-\s]?)?clean\b",
    "Breast-Like":      r"\bbreast[-\s]?like\b",
    "Slow Flow":        r"\bslow\s+flow\b",
    "Medium Flow":      r"\bmedium\s+flow\b",
    "Fast Flow":        r"\bfast\s+flow\b",
    "Silicone Nipple":  r"\bsilicone\s+nipple\b",
    "Wide Neck":        r"\bwide\s+neck\b",
}

LABEL_OVERRIDES: dict = {
    "bpa_free":      "BPA Free",
    "airfree_vent":  "AirFree Vent",
    "air_free":      "Air Free",
    "tommee_tippee": "Tommee Tippee",
    "dr_browns":     "Dr Brown's",
    "dr_brown":      "Dr Brown",
}

BRAND_TOKENS: set = {
    "tommee_tippee", "dr_browns", "dr_brown",
    "philips_avent", "evenflo", "first_years",
    "mam", "nuk", "comotomo", "nanobebe",
    "lansinoh", "medela", "chicco", "playtex", "gerber",
    "sippy_cup",
}

BRAND_NORMALIZATION_MAP: List[Tuple[str, str]] = [
    (r"tommee\s+tippee",       "Tommee Tippee"),
    (r"dr\.?\s+brown'?s?",     "Dr. Brown's"),
    (r"philips\s+avent",       "Philips Avent"),
    (r"evenflo\s+feeding",     "Evenflo"),
    (r"the\s+first\s+years?",  "The First Years"),
    (r"\bfirst\s+years?\b",    "The First Years"),
    (r"\bmam\b",               "MAM"),
    (r"\bnuk\b",               "NUK"),
    (r"comotomo",              "Comotomo"),
    (r"nanobebe",              "Nanobebe"),
    (r"lansinoh",              "Lansinoh"),
    (r"medela",                "Medela"),
    (r"chicco",                "Chicco"),
    (r"playtex",               "Playtex"),
    (r"gerber",                "Gerber"),
]

TITLE_COMPONENTS: dict = {
    "Product Type":  r"\b(?:bottle|bottles?|nipple|nipples?)\b",
    "Flow Rate":     r"\b(?:slow\s+flow|medium\s+flow|fast\s+flow|first\s+flow|variable\s+flow|newborn\s+flow)\b",
    "Anti-Colic":    r"\b(?:anti[-\s]?colic|vented?|vent\s+system|airfree|air[-\s]free)\b",
    "Size / Volume": r"\b\d+\s*(?:oz|ml|fl\.?\s*oz)\b",
    "Quantity":      r"\b(?:\d+\s*[-\u2013]?\s*(?:pack|count|ct\b)|pack\s+of\s+\d+|set\s+of\s+\d+|\d+\s+(?:bottles?|nipples?))\b",
    "Material":      r"\b(?:glass|silicone|stainless|steel|bpa[-\s]?free|ppsu|tritan)\b",
    "Compatibility": r"\b(?:compatible\s+with|works\s+with|fits\s+(?:most|all)?|universal)\b",
}

MESSAGING_CATEGORIES: dict = {
    "Social Proof": [
        r"(?:#\s*1|no\.?\s*1|number\s+one|no\s+1)\s+brand",
        r"(?:recommended\s+by\s+(?:moms?|doctors?|pediatricians?)|doctors?\s+recommended|pediatrician[s]?\s+recommended)",
        r"\bsurvey\b",
        r"\bclinically\s+tested\b",
        r"\baward[- ]?winning\b",
        r"\btrusted\s+by\b",
        r"\bmillion[s]?\s+(of\s+)?moms?\b",
    ],
    "Safety Claims": [
        r"\bbpa[-\s]?free\b",
        r"\bfda\b",
        r"\bnon[-\s]?toxic\b",
        r"\bfood[-\s]?grade\b",
        r"\bhypoallergenic\b",
        r"\bclinically\b",
        r"\bcertified\b",
        r"\bsafe\s+material\b",
    ],
    "Comfort Claims": [
        r"\banti[-\s]?colic\b",
        r"\breflux\b",
        r"\bgas\b",
        r"\bbreast[-\s]?like\b",
        r"\bnatural\s+(response|latch)\b",
        r"\bcomfort\b",
        r"\bsoothe\b",
        r"\bsoothing\b",
        r"\bgentle\b",
    ],
    "Convenience Claims": [
        r"\bdishwasher[-\s]?safe\b",
        r"\beasy[-\s]?(?:to[-\s]?)?clean\b",
        r"\beasy\s+to\s+assemble\b",
        r"\bleak[-\s]?proof\b",
        r"\bno\s+spill\b",
        r"\btravel[-\s]?friendly\b",
        r"\bon[-\s]?the[-\s]?go\b",
        r"\beasy\s+(hold|grip)\b",
    ],
    "Design Claims": [
        r"\bergonomic\b",
        r"\bwide\s+neck\b",
        r"\bsilicone\b",
        r"\bglass\b",
        r"\bstainless\b",
        r"\bcompact\b",
        r"\blightweight\b",
        r"\bgrip\b",
    ],
    "Research / Survey Claims": [
        r"\bclinically\s+tested\b",
        r"\bsurvey\b",
        r"\bresearch\b",
        r"\bstud(?:y|ied)\b",
        r"\bproven\b",
        r"\bexpert\b",
        r"\bpediatrician\b",
        r"\bdentist\b",
        r"\bscientifically\b",
    ],
}

DESC_SECTION_PATTERNS: dict = {
    "Opening Claim": [
        r"\bdesigned\s+(?:for|to)\b", r"\bperfect\s+for\b", r"\bideal\s+for\b",
        r"\bintroducing\b", r"\bthe\s+(?:best|ultimate|only)\b",
    ],
    "Feature Details": [
        r"\bfeatures?\b", r"\bincludes?\b", r"\bequipped\s+with\b",
        r"\bcomes?\s+with\b", r"\bbuilt[-\s]in\b",
    ],
    "Benefit Language": [
        r"\bhelps?\s+\w+", r"\breduces?\b", r"\bprevents?\b",
        r"\bsoothes?\b", r"\bcomfort\b", r"\bpeace\s+of\s+mind\b", r"\bno\s+more\b",
    ],
    "Safety & Materials": [
        r"\bbpa[-\s]?free\b", r"\bfood[-\s]grade\b", r"\bnon[-\s]?toxic\b",
        r"\bfda\b", r"\bhypoallergenic\b",
    ],
    "Age & Fit": [
        r"\b(?:newborn|infant|0\+?\s*m(?:onths?)?|\d+\+?\s*m(?:onths?)?)\b",
        r"\bsuitable\s+for\b", r"\bcompatible\s+with\b", r"\bworks\s+with\b",
    ],
    "Care & Usage": [
        r"\bdishwasher[-\s]?safe\b", r"\beasy[-\s](?:to[-\s])?clean\b",
        r"\bsteriliz\w+\b", r"\bhow\s+to\s+use\b", r"\bwash\s+before\b",
    ],
    "Trust Signals": [
        r"(?:#\s*1|no\.?\s*1|number\s+one|no\s+1)\s+brand",
        r"\brecommended\s+by\b", r"\bclinically\b",
        r"\baward\b", r"\btrusted\b", r"\bmillion\w*\s+(?:of\s+)?moms?\b",
        r"\bpediatrician\b",
    ],
}

FEATURE_LANGUAGE_PATTERNS: List[str] = [
    r"\bmade\s+(?:of|from|with)\b",
    r"\bconstructed\s+(?:of|from)\b",
    r"\bfeatures?\b",
    r"\bincludes?\b",
    r"\bequipped\s+with\b",
    r"\bcomes?\s+with\b",
    r"\bbuilt[-\s]in\b",
    r"\bmaterial[s]?\b",
    r"\bcomponent[s]?\b",
    r"\bdimension[s]?\b",
    r"\b\d+\s*(?:oz|ml)\b",
    r"\b\d+\s*[-\u2013]?\s*(?:pack|count)\b",
]

BENEFIT_LANGUAGE_PATTERNS: List[str] = [
    r"\bhelps?\s+(?:reduce|prevent|soothe|calm|ease|avoid|eliminate)\b",
    r"\breduces?\b",
    r"\bprevents?\b",
    r"\bsoothes?\b",
    r"\bcomfort\b",
    r"\bso\s+(?:you|baby|your\s+baby)\s+can\b",
    r"\bgiving\s+(?:your|baby)\b",
    r"\bpeace\s+of\s+mind\b",
    r"\bworry[-\s]free\b",
    r"\bno\s+more\b",
    r"\bmakes?\s+it\s+(?:easy|easier|simple)\b",
    r"\bperfect\s+for\b",
    r"\bideal\s+for\b",
    r"\bdesigned\s+to\s+(?:help|reduce|prevent|ease|support)\b",
    r"\bensures?\b",
    r"\bpromotes?\b",
    r"\bencourages?\b",
    r"\bprovides?\s+(?:comfort|relief|support)\b",
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def normalize_brand(raw_brand: str) -> str:
    """Return a canonical brand name from a raw brand string."""
    if not isinstance(raw_brand, str) or not raw_brand.strip():
        return "Unknown"
    b = raw_brand.strip()
    for pattern, canonical in BRAND_NORMALIZATION_MAP:
        if re.search(pattern, b, flags=re.IGNORECASE):
            return canonical
    return b.title() if b else "Unknown"


def token_to_label(token: str) -> str:
    """Convert an underscore token to a human-readable display label."""
    if token in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[token]
    return token.replace("_", " ").title()


def normalize_text_for_phrases(text: str, phrase_map: List[Tuple[str, str]]) -> str:
    """Replace multi-word phrases with underscore tokens before tokenization."""
    normalized = text.lower()
    for pattern, token in phrase_map:
        normalized = re.sub(pattern, token, normalized, flags=re.IGNORECASE)
    return normalized


def extract_keyword_frequencies(
    series: pd.Series,
    phrase_map: List[Tuple[str, str]],
    stopwords: set,
    top_n: int = 20,
) -> pd.DataFrame:
    """Normalize → tokenize → filter → count; return top-N as DataFrame."""
    all_tokens: List[str] = []
    for text in series.dropna().astype(str):
        normalized = normalize_text_for_phrases(text, phrase_map)
        raw_tokens = re.findall(r"[a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)+|[a-z][a-z0-9]*", normalized)
        for tok in raw_tokens:
            if "_" in tok:
                if tok not in BRAND_TOKENS:
                    all_tokens.append(tok)
            elif tok not in stopwords and len(tok) > 2:
                all_tokens.append(tok)
    freq_pairs = Counter(all_tokens).most_common(top_n)
    rows = [(token_to_label(tok), cnt) for tok, cnt in freq_pairs]
    return pd.DataFrame(rows, columns=["keyword", "frequency"])


def build_claim_presence(
    df: pd.DataFrame, text_col: str, claim_map: dict
) -> pd.DataFrame:
    """Count listings containing each claim (binary — each listing counted once)."""
    text_series = df[text_col].fillna("").astype(str)
    total = len(df)
    results = []
    for label, pattern in claim_map.items():
        count = int(text_series.str.contains(pattern, flags=re.IGNORECASE, regex=True).sum())
        pct = round(count / total * 100, 1) if total > 0 else 0.0
        results.append({"claim": label, "listings": count, "% of Products": f"{pct}%"})
    return (
        pd.DataFrame(results)
        .sort_values("listings", ascending=False)
        .reset_index(drop=True)
    )


def build_keyword_gap_table(
    tt_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    text_col: str,
    phrase_map: List[Tuple[str, str]],
    stopwords: set,
    top_n: int = 40,
) -> pd.DataFrame:
    """Compare keyword % between TT and competitors (normalized by group size)."""
    tt_kw   = extract_keyword_frequencies(tt_df[text_col].dropna(),   phrase_map, stopwords, top_n=top_n)
    comp_kw = extract_keyword_frequencies(comp_df[text_col].dropna(), phrase_map, stopwords, top_n=top_n)
    tt_n   = max(len(tt_df), 1)
    comp_n = max(len(comp_df), 1)
    tt_kw   = tt_kw.copy()
    tt_kw["TT %"]    = (tt_kw["frequency"]   / tt_n   * 100).round(1)
    comp_kw = comp_kw.copy()
    comp_kw["Comp %"] = (comp_kw["frequency"] / comp_n * 100).round(1)
    merged = pd.merge(
        tt_kw.rename(columns={"frequency": "TT Count"})[["keyword", "TT Count", "TT %"]],
        comp_kw.rename(columns={"frequency": "Comp Count"})[["keyword", "Comp Count", "Comp %"]],
        on="keyword", how="outer",
    ).fillna(0)
    merged["TT Count"]   = merged["TT Count"].astype(int)
    merged["Comp Count"] = merged["Comp Count"].astype(int)
    merged["Gap % (Comp \u2212 TT)"] = (merged["Comp %"] - merged["TT %"]).round(1)
    return (
        merged.rename(columns={"keyword": "Keyword / Phrase"})
        [["Keyword / Phrase", "TT Count", "TT %", "Comp Count", "Comp %", "Gap % (Comp \u2212 TT)"]]
        .sort_values("Gap % (Comp \u2212 TT)", ascending=False)
        .reset_index(drop=True)
    )


def build_claim_gap_table(
    tt_df: pd.DataFrame, comp_df: pd.DataFrame, text_col: str, claim_map: dict
) -> pd.DataFrame:
    """Compare marketing claim presence (%) between TT and competitors."""
    tt_n   = max(len(tt_df), 1)
    comp_n = max(len(comp_df), 1)
    tt_raw   = build_claim_presence(tt_df,   text_col, claim_map)[["claim", "listings"]].rename(columns={"listings": "TT Listings"})
    comp_raw = build_claim_presence(comp_df, text_col, claim_map)[["claim", "listings"]].rename(columns={"listings": "Comp Listings"})
    merged = pd.merge(tt_raw, comp_raw, on="claim", how="outer").fillna(0)
    merged["TT Listings"]   = merged["TT Listings"].astype(int)
    merged["Comp Listings"] = merged["Comp Listings"].astype(int)
    merged["TT %"]   = (merged["TT Listings"]   / tt_n   * 100).round(1)
    merged["Comp %"] = (merged["Comp Listings"] / comp_n * 100).round(1)
    merged["Gap % (Comp \u2212 TT)"] = (merged["Comp %"] - merged["TT %"]).round(1)
    return (
        merged[["claim", "TT Listings", "TT %", "Comp Listings", "Comp %", "Gap % (Comp \u2212 TT)"]]
        .sort_values("Gap % (Comp \u2212 TT)", ascending=False)
        .reset_index(drop=True)
    )


def build_messaging_comparison(
    df_a: pd.DataFrame, df_b: pd.DataFrame, text_col: str,
    label_a: str = "Group A", label_b: str = "Group B",
) -> pd.DataFrame:
    """Return long-format DataFrame of messaging category % for two groups."""
    rows = []
    for group_df, label in [(df_a, label_a), (df_b, label_b)]:
        n = max(len(group_df), 1)
        text_series = group_df[text_col].fillna("").astype(str)
        for category, patterns in MESSAGING_CATEGORIES.items():
            combined = "|".join(f"(?:{p})" for p in patterns)
            count = text_series.str.contains(combined, flags=re.IGNORECASE, regex=True).sum()
            rows.append({
                "Category": category,
                "Group": label,
                "% of Listings": round(count / n * 100, 1),
            })
    return pd.DataFrame(rows)


def render_image_grid(
    img_df: pd.DataFrame, img_col: str, images_per_row: int = 4
) -> None:
    """Render product thumbnails in a grid with brand and title captions."""
    for row_start in range(0, len(img_df), images_per_row):
        row_slice = img_df.iloc[row_start: row_start + images_per_row]
        cols = st.columns(images_per_row)
        for col, (_, product) in zip(cols, row_slice.iterrows()):
            with col:
                try:
                    st.image(str(product[img_col]), use_container_width=True)
                except Exception:
                    st.write("Image unavailable")
                brand_label = str(product.get("brand", "")) if pd.notna(product.get("brand")) else ""
                raw_title = str(product.get("title", ""))
                title_label = raw_title[:55] + "\u2026" if len(raw_title) > 55 else raw_title
                st.caption(f"**{brand_label}**  \n{title_label}")


def filter_bottles_and_nipples(df: pd.DataFrame) -> pd.DataFrame:
    """Return only baby bottle and nipple listings (excludes sippy cups, warmers, etc.)."""
    if df.empty:
        return df
    title_lc = df.get("title", pd.Series(dtype=str)).fillna("").str.lower()
    bc_lc    = df.get("breadcrumbs", pd.Series(dtype=str)).fillna("").str.lower()
    is_target = (
        title_lc.str.contains(r"\b(?:bottle|nipple)\b", regex=True)
        | bc_lc.str.contains(r"\b(?:bottle|nipple)\b", regex=True)
    )
    is_excluded = (
        title_lc.str.contains(r"\b(?:sippy|straw\s+cup|warmer|steriliz|pacifier|trainer\s+cup)\b", regex=True)
        | bc_lc.str.contains(r"\b(?:sippy|warmer|steriliz|pacifier)\b", regex=True)
    )
    return df[is_target & ~is_excluded].copy()


def compute_title_quality_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Score each title 0–100 across TITLE_COMPONENTS (7 binary components)."""
    if "title" not in df.columns or df.empty:
        return pd.DataFrame()
    text = df["title"].fillna("").astype(str)
    result = pd.DataFrame(index=df.index)
    for component, pattern in TITLE_COMPONENTS.items():
        result[component] = text.str.contains(pattern, flags=re.IGNORECASE, regex=True).astype(int)
    result["title_quality_score"] = (
        result[list(TITLE_COMPONENTS.keys())].sum(axis=1) / len(TITLE_COMPONENTS) * 100
    ).round(1)
    if "brand" in df.columns:
        result["brand"] = df["brand"].values
    return result


def compute_description_quality_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Score each description 0–100 across DESC_SECTION_PATTERNS (7 sections)."""
    if "description" not in df.columns or df.empty:
        return pd.DataFrame()
    text = df["description"].fillna("").astype(str)
    result = pd.DataFrame(index=df.index)
    for section, patterns in DESC_SECTION_PATTERNS.items():
        combined = "|".join(f"(?:{p})" for p in patterns)
        result[section] = text.str.contains(combined, flags=re.IGNORECASE, regex=True).astype(int)
    result["description_quality_score"] = (
        result[list(DESC_SECTION_PATTERNS.keys())].sum(axis=1) / len(DESC_SECTION_PATTERNS) * 100
    ).round(1)
    if "brand" in df.columns:
        result["brand"] = df["brand"].values
    return result


def compute_feature_benefit_balance(
    df: pd.DataFrame, text_col: str = "description"
) -> pd.DataFrame:
    """Classify each listing as Feature-Led / Benefit-Led / Balanced / No Signal."""
    if text_col not in df.columns or df.empty:
        return pd.DataFrame()
    text = df[text_col].fillna("").astype(str)
    result = pd.DataFrame(index=df.index)
    result["feature_hits"] = sum(
        text.str.contains(p, flags=re.IGNORECASE, regex=True).astype(int)
        for p in FEATURE_LANGUAGE_PATTERNS
    )
    result["benefit_hits"] = sum(
        text.str.contains(p, flags=re.IGNORECASE, regex=True).astype(int)
        for p in BENEFIT_LANGUAGE_PATTERNS
    )

    def _classify(row: pd.Series) -> str:
        f, b = row["feature_hits"], row["benefit_hits"]
        if f == 0 and b == 0:
            return "No Signal"
        if f > b * 1.5:
            return "Feature-Led"
        if b > f * 1.5:
            return "Benefit-Led"
        return "Balanced"

    result["orientation"] = result.apply(_classify, axis=1)
    if "brand" in df.columns:
        result["brand"] = df["brand"].values
    return result


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    """Load the cleaned CSV, normalise types and brands, filter to baby products."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ["stars", "reviewsCount", "price_value", "title_word_count", "description_length"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "thumbnailimage" in df.columns:
        df.rename(columns={"thumbnailimage": "thumbnailImage"}, inplace=True)
    if "brand" in df.columns:
        df["brand"] = df["brand"].apply(normalize_brand)
    bc_col = None
    for candidate in ["breadcrumbs", "breadCrumbs", "Breadcrumbs"]:
        if candidate in df.columns:
            bc_col = candidate
            break
    if bc_col is not None:
        is_baby = df[bc_col].astype(str).str.contains("baby", case=False, na=True)
        df = df[is_baby].copy()
    if "price_value" in df.columns:
        df = df[df["price_value"].isna() | df["price_value"].between(3, 100)].copy()
    return df


@st.cache_data
def load_tt_immersion_data() -> pd.DataFrame:
    """Load the Tommee Tippee immersion CSV (3 branded search queries)."""
    if not TT_IMMERSION_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(TT_IMMERSION_FILE)
    for col in ["stars", "reviewsCount", "price_value", "title_word_count",
                "description_length", "popularity_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for bool_col in ["is_tommee_tippee_brand", "is_competitor_brand",
                     "is_compatible_accessory", "is_core_bottle_listing"]:
        if bool_col in df.columns:
            df[bool_col] = df[bool_col].astype(bool)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR FILTERS
# ══════════════════════════════════════════════════════════════════════════════

def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters and return the filtered DataFrame."""
    st.sidebar.header("Filters")
    filtered = df.copy()
    if "brand" in df.columns:
        brand_options = sorted(df["brand"].dropna().astype(str).unique())
        selected_brands = st.sidebar.multiselect("Brand", brand_options, default=brand_options)
        filtered = filtered[filtered["brand"].astype(str).isin(selected_brands)]
    if "price_value" in df.columns:
        price_clean = df["price_value"].dropna()
        if not price_clean.empty:
            min_p, max_p = float(price_clean.min()), float(price_clean.max())
            if min_p < max_p:
                price_range = st.sidebar.slider(
                    "Price Range ($)", min_value=min_p, max_value=max_p,
                    value=(min_p, max_p), step=0.5,
                )
                filtered = filtered[
                    filtered["price_value"].isna()
                    | filtered["price_value"].between(price_range[0], price_range[1])
                ]
    if "stars" in df.columns:
        rating_range = st.sidebar.slider(
            "Rating Range", min_value=0.0, max_value=5.0,
            value=(0.0, 5.0), step=0.1,
        )
        filtered = filtered[
            filtered["stars"].isna()
            | filtered["stars"].between(rating_range[0], rating_range[1])
        ]
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# SHARED: TT / COMPETITOR SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def _split_tt_comp(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Split df into (tt_df, comp_df). Returns (tt, comp, tt_found).

    tt_found=False means Tommee Tippee was filtered out; caller should warn.
    """
    if "brand" not in df.columns:
        return pd.DataFrame(), df.copy(), False
    is_tt = df["brand"].astype(str).str.contains("tommee", case=False, na=False)
    return df[is_tt].copy(), df[~is_tt].copy(), bool(is_tt.any())


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def tab_overview(df: pd.DataFrame) -> None:
    st.header("Market Overview")
    st.caption("The competitive landscape across baby bottle listings in the filtered dataset.")

    # ── KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Listings", f"{len(df):,}")
    avg_rating = df["stars"].mean() if "stars" in df.columns else np.nan
    c2.metric("Avg Rating", f"{avg_rating:.2f} \u2605" if pd.notna(avg_rating) else "N/A")
    avg_price = df["price_value"].mean() if "price_value" in df.columns else np.nan
    c3.metric("Avg Price", f"${avg_price:.2f}" if pd.notna(avg_price) else "N/A")
    n_brands = df["brand"].nunique() if "brand" in df.columns else 0
    c4.metric("Brands Competing", f"{n_brands:,}")

    if "brand" not in df.columns:
        st.info("No brand column found.")
        return

    # ── Tommee Tippee position callout
    is_tt = df["brand"].astype(str).str.contains("tommee", case=False, na=False)
    tt_count = int(is_tt.sum())
    if tt_count > 0:
        tt_pct = round(tt_count / max(len(df), 1) * 100)
        tt_avg_r = df.loc[is_tt, "stars"].mean() if "stars" in df.columns else np.nan
        tt_avg_p = df.loc[is_tt, "price_value"].mean() if "price_value" in df.columns else np.nan
        parts = [f"**Tommee Tippee** has **{tt_count} listings** in this dataset ({tt_pct}% of shelf)"]
        if pd.notna(tt_avg_r):
            parts.append(f"avg rating **{tt_avg_r:.1f} \u2605**")
        if pd.notna(tt_avg_p):
            parts.append(f"avg price **${tt_avg_p:.2f}**")
        st.info("  ·  ".join(parts))

    st.divider()

    # ── Brand charts
    st.subheader("Brand Landscape")
    st.caption("Brands ranked by listing count and by total customer review volume. Tommee Tippee is highlighted in blue.")

    left, right = st.columns(2)

    with left:
        brand_count = (
            df["brand"].value_counts().head(12)
            .rename_axis("brand").reset_index(name="listings")
        )
        brand_count["color"] = brand_count["brand"].apply(
            lambda b: TT_BLUE if "tommee" in b.lower() else "#aaaaaa"
        )
        fig = px.bar(
            brand_count, x="listings", y="brand", orientation="h",
            title="Listings per Brand (Top 12)",
            color="brand",
            color_discrete_map={b: c for b, c in zip(brand_count["brand"], brand_count["color"])},
        )
        fig.update_layout(showlegend=False, yaxis={"autorange": "reversed"},
                          xaxis_title="Listings", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if "reviewsCount" in df.columns:
            brand_rev = (
                df.groupby("brand", dropna=False)["reviewsCount"]
                .sum().sort_values(ascending=False).head(12).reset_index()
            )
            brand_rev["color"] = brand_rev["brand"].apply(
                lambda b: TT_BLUE if "tommee" in str(b).lower() else "#aaaaaa"
            )
            fig2 = px.bar(
                brand_rev, x="reviewsCount", y="brand", orientation="h",
                title="Total Reviews per Brand (Top 12)",
                color="brand",
                color_discrete_map={b: c for b, c in zip(brand_rev["brand"], brand_rev["color"])},
            )
            fig2.update_layout(showlegend=False, yaxis={"autorange": "reversed"},
                               xaxis_title="Total Reviews", yaxis_title="")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("reviewsCount column not available.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: TITLE INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

def tab_title_intelligence(df: pd.DataFrame) -> None:
    st.header("Title Intelligence")
    st.caption("How Tommee Tippee and competitors write their product titles. Scoped to baby bottles and nipples.")

    df = filter_bottles_and_nipples(df)
    if df.empty:
        st.warning("No bottle or nipple listings found with the current filters.")
        return

    tt_df, comp_df, tt_found = _split_tt_comp(df)
    if not tt_found:
        st.warning(
            "Tommee Tippee listings not found in the current filtered data. "
            "Re-enable Tommee Tippee in the sidebar Brand filter to see this analysis."
        )
        return

    title_col = "clean_title" if "clean_title" in df.columns else "title"
    st.caption(
        f"Comparing **{len(tt_df):,} Tommee Tippee** listings against "
        f"**{len(comp_df):,} competitor** listings (after sidebar filters)."
    )

    # ── Section 1: Keywords side by side
    st.subheader("Top Title Keywords")
    st.caption("Most frequent meaningful words and phrases in product titles (brand names excluded).")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Tommee Tippee** ({len(tt_df)} listings)")
        if title_col in tt_df.columns:
            tt_kw = extract_keyword_frequencies(tt_df[title_col], TITLE_PHRASE_MAP, TITLE_STOPWORDS, top_n=15)
            if not tt_kw.empty:
                fig = px.bar(
                    tt_kw, x="frequency", y="keyword", orientation="h",
                    color_discrete_sequence=[TT_BLUE],
                )
                fig.update_layout(showlegend=False, yaxis={"autorange": "reversed"},
                                  xaxis_title="Frequency", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No keywords found.")
        else:
            st.info("Title column not available.")

    with col2:
        st.markdown(f"**Competitors** ({len(comp_df)} listings)")
        if not comp_df.empty and title_col in comp_df.columns:
            comp_kw = extract_keyword_frequencies(comp_df[title_col], TITLE_PHRASE_MAP, TITLE_STOPWORDS, top_n=15)
            if not comp_kw.empty:
                fig2 = px.bar(
                    comp_kw, x="frequency", y="keyword", orientation="h",
                    color_discrete_sequence=[COMP_ORANGE],
                )
                fig2.update_layout(showlegend=False, yaxis={"autorange": "reversed"},
                                   xaxis_title="Frequency", yaxis_title="")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No keywords found.")
        elif comp_df.empty:
            st.info("No competitor listings in current filter.")

    st.divider()

    # ── Section 2: Title attribute coverage
    st.subheader("Title Attribute Coverage")
    st.caption(
        "What percentage of titles include each key product attribute? "
        "Gaps between the bars show where competitors mention attributes Tommee Tippee does not — and vice versa."
    )

    if not comp_df.empty and "title" in df.columns:
        coverage_rows = []
        for component, pattern in TITLE_COMPONENTS.items():
            tt_hits   = tt_df["title"].dropna().str.contains(pattern, case=False, regex=True).sum()
            comp_hits = comp_df["title"].dropna().str.contains(pattern, case=False, regex=True).sum()
            coverage_rows.append({
                "Attribute":       component,
                "Tommee Tippee":   round(tt_hits   / max(len(tt_df),   1) * 100, 1),
                "Competitors":     round(comp_hits / max(len(comp_df), 1) * 100, 1),
            })
        cov_df = pd.DataFrame(coverage_rows)
        fig_cov = px.bar(
            cov_df.melt(id_vars="Attribute", var_name="Brand Group", value_name="% of Titles"),
            x="Attribute", y="% of Titles", color="Brand Group", barmode="group",
            color_discrete_map={"Tommee Tippee": TT_BLUE, "Competitors": COMP_ORANGE},
        )
        fig_cov.update_layout(xaxis_title="", legend_title="")
        st.plotly_chart(fig_cov, use_container_width=True)
    else:
        st.info("Competitor data or title column not available for this chart.")

    st.divider()

    # ── Section 3: Title Quality Score
    st.subheader("Title Quality Score")
    st.caption(
        "Each title is scored on how many of 7 key attributes it includes: "
        "Product Type, Flow Rate, Anti-Colic, Size/Volume, Quantity, Material, Compatibility."
    )
    tt_scores   = compute_title_quality_scores(tt_df)
    comp_scores = compute_title_quality_scores(comp_df)
    if not tt_scores.empty and not comp_scores.empty:
        tt_avg   = tt_scores["title_quality_score"].mean()
        comp_avg = comp_scores["title_quality_score"].mean()
        delta    = tt_avg - comp_avg
        c1, c2, c3 = st.columns(3)
        c1.metric("Tommee Tippee Avg", f"{tt_avg:.0f} / 100")
        c2.metric("Competitor Avg",    f"{comp_avg:.0f} / 100")
        c3.metric("TT vs Competitors", f"{delta:+.0f} pts",
                  help="Positive = TT titles are more complete. Negative = competitors lead.")
    elif tt_scores.empty:
        st.info("Title quality scores could not be computed.")

    st.divider()

    # ── Section 4: Top TT title examples
    st.subheader("Top Tommee Tippee Titles")
    st.caption("Highest-reviewed Tommee Tippee listings sorted by review count.")
    if "title" in tt_df.columns:
        ex_cols = [c for c in ["title", "reviewsCount", "stars", "price_value"] if c in tt_df.columns]
        tt_examples = (
            tt_df[ex_cols].dropna(subset=["title"])
            .sort_values("reviewsCount", ascending=False)
            .head(12).reset_index(drop=True)
        )
        col_cfg: dict = {}
        if "stars" in tt_examples.columns:
            col_cfg["stars"] = st.column_config.NumberColumn("Rating", format="%.1f \u2605")
        if "reviewsCount" in tt_examples.columns:
            col_cfg["reviewsCount"] = st.column_config.NumberColumn("Reviews", format="%d")
        if "price_value" in tt_examples.columns:
            col_cfg["price_value"] = st.column_config.NumberColumn("Price", format="$%.2f")
        st.dataframe(tt_examples, column_config=col_cfg, use_container_width=True, hide_index=True)
    else:
        st.info("Title column not available.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: MESSAGING & CLAIMS
# ══════════════════════════════════════════════════════════════════════════════

def tab_messaging(df: pd.DataFrame) -> None:
    st.header("Messaging & Claims")
    st.caption("What language brands use in product descriptions. Scoped to baby bottles and nipples.")

    df = filter_bottles_and_nipples(df)
    if df.empty:
        st.warning("No bottle or nipple listings found with the current filters.")
        return

    tt_df, comp_df, tt_found = _split_tt_comp(df)
    if not tt_found:
        st.warning(
            "Tommee Tippee listings not found in the current filtered data. "
            "Re-enable Tommee Tippee in the sidebar Brand filter to see this analysis."
        )
        return

    if "description" not in df.columns:
        st.info("No description data available in this dataset.")
        return

    st.caption(
        f"Comparing **{len(tt_df):,} Tommee Tippee** listings against "
        f"**{len(comp_df):,} competitor** listings (after sidebar filters)."
    )

    # ── Section 1: Marketing claims comparison
    st.subheader("Marketing Claims — Tommee Tippee vs Competitors")
    st.caption("% of listings that contain each claim. Each listing counted once per claim regardless of repetition.")

    if not comp_df.empty:
        claim_gap = build_claim_gap_table(tt_df, comp_df, "description", CLAIM_MAP)
        claims_long = pd.concat([
            claim_gap[["claim", "TT %"]].rename(columns={"TT %": "% of Listings"}).assign(Group="Tommee Tippee"),
            claim_gap[["claim", "Comp %"]].rename(columns={"Comp %": "% of Listings"}).assign(Group="Competitors"),
        ]).reset_index(drop=True)
        fig_claims = px.bar(
            claims_long, x="% of Listings", y="claim", color="Group",
            barmode="group", orientation="h",
            color_discrete_map={"Tommee Tippee": TT_BLUE, "Competitors": COMP_ORANGE},
            title="Marketing Claim Usage — % of Listings",
        )
        fig_claims.update_layout(yaxis={"autorange": "reversed"}, legend_title="")
        st.plotly_chart(fig_claims, use_container_width=True)

        st.caption(
            "Gap column shows Competitor % minus Tommee Tippee %. "
            "Positive values mean competitors use that claim more often."
        )
        st.dataframe(
            claim_gap.rename(columns={"claim": "Claim"}),
            use_container_width=True, hide_index=True,
        )
    else:
        # No competitors — show TT claims alone
        tt_claims = build_claim_presence(tt_df, "description", CLAIM_MAP)
        active = tt_claims[tt_claims["listings"] > 0]
        if not active.empty:
            fig = px.bar(active, x="claim", y="listings",
                         title="Tommee Tippee Marketing Claims",
                         color_discrete_sequence=[TT_BLUE])
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Section 2: Messaging category coverage
    st.subheader("Messaging Category Coverage")
    st.caption(
        "Six broad categories detected in descriptions. "
        "Shows what share of each brand group's listings cover each category."
    )
    if not comp_df.empty:
        msg_cmp = build_messaging_comparison(tt_df, comp_df, "description", "Tommee Tippee", "Competitors")
        fig_msg = px.bar(
            msg_cmp, x="Category", y="% of Listings", color="Group",
            barmode="group",
            color_discrete_map={"Tommee Tippee": TT_BLUE, "Competitors": COMP_ORANGE},
        )
        fig_msg.update_layout(xaxis_title="", legend_title="")
        st.plotly_chart(fig_msg, use_container_width=True)

    st.divider()

    # ── Section 3: Feature vs Benefit language
    st.subheader("Feature vs Benefit Language")
    st.caption(
        "Are descriptions written around product specs (Feature-Led), "
        "customer outcomes (Benefit-Led), or a mix (Balanced)?"
    )
    tt_fb   = compute_feature_benefit_balance(tt_df)
    comp_fb = compute_feature_benefit_balance(comp_df)

    if not tt_fb.empty and not comp_fb.empty:
        orientation_order = ["Feature-Led", "Balanced", "Benefit-Led", "No Signal"]
        tt_counts   = tt_fb["orientation"].value_counts().reindex(orientation_order, fill_value=0)
        comp_counts = comp_fb["orientation"].value_counts().reindex(orientation_order, fill_value=0)
        fb_df = pd.DataFrame({
            "Orientation":   orientation_order,
            "Tommee Tippee": (tt_counts   / max(len(tt_fb),   1) * 100).round(1).values,
            "Competitors":   (comp_counts / max(len(comp_fb), 1) * 100).round(1).values,
        })
        fb_long = fb_df.melt(id_vars="Orientation", var_name="Group", value_name="% of Listings")
        fig_fb = px.bar(
            fb_long, x="Orientation", y="% of Listings", color="Group",
            barmode="group",
            color_discrete_map={"Tommee Tippee": TT_BLUE, "Competitors": COMP_ORANGE},
        )
        fig_fb.update_layout(xaxis_title="", legend_title="")
        st.plotly_chart(fig_fb, use_container_width=True)
    elif tt_fb.empty:
        st.info("Feature/benefit balance could not be computed.")

    st.divider()

    # ── Section 4: Description Quality Score
    st.subheader("Description Quality Score")
    st.caption(
        "Each description is scored on how many of 7 content sections it covers: "
        "Opening Claim, Feature Details, Benefit Language, Safety & Materials, "
        "Age & Fit, Care & Usage, Trust Signals."
    )
    tt_dq   = compute_description_quality_scores(tt_df)
    comp_dq = compute_description_quality_scores(comp_df)
    if not tt_dq.empty and not comp_dq.empty:
        tt_avg   = tt_dq["description_quality_score"].mean()
        comp_avg = comp_dq["description_quality_score"].mean()
        delta    = tt_avg - comp_avg
        c1, c2, c3 = st.columns(3)
        c1.metric("Tommee Tippee Avg", f"{tt_avg:.0f} / 100")
        c2.metric("Competitor Avg",    f"{comp_avg:.0f} / 100")
        c3.metric("TT vs Competitors", f"{delta:+.0f} pts",
                  help="Positive = TT descriptions cover more sections. Negative = competitors lead.")
    elif tt_dq.empty:
        st.info("Description quality scores could not be computed.")

    st.divider()

    # ── Section 5: Messaging Theme Gap table
    st.subheader("Messaging Theme Gap")
    st.caption(
        "Phrases appearing more often in competitor descriptions than in Tommee Tippee's. "
        "Positive gap values indicate themes where competitors are more active."
    )
    if not comp_df.empty:
        desc_gap = build_keyword_gap_table(
            tt_df, comp_df, "description", DESC_PHRASE_MAP, DESC_STOPWORDS, top_n=30
        )
        desc_gap_display = desc_gap[
            (desc_gap["TT Count"] > 0) | (desc_gap["Comp Count"] > 0)
        ].head(20)
        if not desc_gap_display.empty:
            st.dataframe(desc_gap_display, use_container_width=True, hide_index=True)
        else:
            st.info("No shared messaging themes found for comparison.")
    else:
        st.info("No competitor listings in current filter for gap analysis.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: BRANDED SEARCH (TT Immersion)
# Independent dataset — does NOT use sidebar-filtered df.
# ══════════════════════════════════════════════════════════════════════════════

def tab_branded_search() -> None:
    """Tommee Tippee branded search competitive landscape."""
    df_full = load_tt_immersion_data()
    if df_full.empty:
        st.error(
            "Tommee Tippee branded search dataset not found. "
            "Run `scripts/clean_tt_immersion.py` to generate it."
        )
        return

    st.header("Branded Search")
    st.subheader("Competitive Landscape in Tommee Tippee Search Results")

    st.info(
        "**Data Scope**\n\n"
        "This tab only represents results for these 3 branded keywords:\n\n"
        "- tommee tippee bottle\n"
        "- tommee tippee anti colic bottles\n"
        "- tommee tippee bottle newborn\n\n"
        "The analysis shows the competitive landscape **within Tommee Tippee branded search results**, including:\n\n"
        "- Tommee Tippee listings\n"
        "- Competitor listings appearing in those searches\n"
        "- Accessory and compatible products that surface in those results\n\n"
        "This section does **NOT** represent the entire baby bottle category. "
        "It only reflects what shoppers see when they search for Tommee Tippee-related queries."
    )

    # ── Inline filters
    with st.expander("Filter this view", expanded=False):
        col_f1, col_f2, col_f3 = st.columns(3)

        all_types = sorted(df_full["product_type"].dropna().unique().tolist()) if "product_type" in df_full.columns else []
        selected_types = col_f1.multiselect("Product Type", all_types, default=all_types, key="cl_tt_types")

        price_min = float(df_full["price_value"].min(skipna=True)) if "price_value" in df_full.columns and df_full["price_value"].notna().any() else 0.0
        price_max = float(df_full["price_value"].max(skipna=True)) if "price_value" in df_full.columns and df_full["price_value"].notna().any() else 200.0
        price_range = col_f2.slider("Price range ($)", min_value=price_min, max_value=price_max,
                                    value=(price_min, price_max), step=0.5, key="cl_tt_price")

        show_only = col_f3.radio(
            "Show", ["All listings", "Tommee Tippee only", "Competitors only"],
            key="cl_tt_show"
        )

    # Apply inline filters
    df = df_full.copy()
    if selected_types and "product_type" in df.columns:
        df = df[df["product_type"].isin(selected_types)]
    if "price_value" in df.columns:
        df = df[df["price_value"].isna() | df["price_value"].between(price_range[0], price_range[1])]
    if show_only == "Tommee Tippee only" and "is_tommee_tippee_brand" in df.columns:
        df = df[df["is_tommee_tippee_brand"]]
    elif show_only == "Competitors only" and "is_competitor_brand" in df.columns:
        df = df[df["is_competitor_brand"]]

    if df.empty:
        st.warning("No listings match the current filters.")
        return

    tt   = df[df["is_tommee_tippee_brand"]] if "is_tommee_tippee_brand" in df.columns else pd.DataFrame()
    comp = df[df["is_competitor_brand"]]    if "is_competitor_brand"    in df.columns else pd.DataFrame()
    total_listings = len(df)

    st.divider()

    # ── Share of Shelf
    st.subheader("Share of Shelf")
    st.caption("How many listings each brand owns within these Tommee Tippee-branded search results.")

    tt_count   = int(df["is_tommee_tippee_brand"].sum()) if "is_tommee_tippee_brand" in df.columns else 0
    comp_count = int(df["is_competitor_brand"].sum())    if "is_competitor_brand"    in df.columns else 0
    other_count = total_listings - tt_count - comp_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Tommee Tippee",       f"{tt_count}",   f"{tt_count/max(total_listings,1)*100:.0f}% of shelf")
    col2.metric("Competitors",         f"{comp_count}", f"{comp_count/max(total_listings,1)*100:.0f}% of shelf")
    col3.metric("Other / Accessories", f"{other_count}")

    if "brand" in df.columns:
        brand_counts = df["brand"].value_counts().head(15).reset_index()
        brand_counts.columns = ["Brand", "Listings"]
        brand_counts["color"] = brand_counts["Brand"].apply(
            lambda b: TT_BLUE if "tommee" in b.lower() else "#aaaaaa"
        )
        fig_shelf = px.bar(
            brand_counts, x="Listings", y="Brand", orientation="h",
            color="Brand",
            color_discrete_map={b: c for b, c in zip(brand_counts["Brand"], brand_counts["color"])},
            title="Top Brands by Listing Count",
        )
        fig_shelf.update_layout(showlegend=False, yaxis={"autorange": "reversed"})
        st.plotly_chart(fig_shelf, use_container_width=True)

    st.divider()

    # ── Competitor Interception
    st.subheader("Competitor Interception")
    st.caption(
        "Which competitors appear most often in these Tommee Tippee search results? "
        "High presence signals direct competition for these search terms."
    )

    if not comp.empty and "brand" in comp.columns:
        comp_brands = comp["brand"].value_counts().reset_index()
        comp_brands.columns = ["Competitor Brand", "Listings"]
        fig_comp = px.bar(
            comp_brands, x="Listings", y="Competitor Brand", orientation="h",
            color_discrete_sequence=[COMP_ORANGE],
            title="Competitor Brands — Listing Count in TT Search Space",
        )
        fig_comp.update_layout(yaxis={"autorange": "reversed"})
        st.plotly_chart(fig_comp, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        tt_avg_r    = tt["stars"].mean()       if not tt.empty   and "stars"        in tt.columns   else None
        comp_avg_r  = comp["stars"].mean()     if not comp.empty and "stars"        in comp.columns else None
        tt_avg_rev  = tt["reviewsCount"].mean()   if not tt.empty   and "reviewsCount" in tt.columns   else None
        comp_avg_rev = comp["reviewsCount"].mean() if not comp.empty and "reviewsCount" in comp.columns else None

        c1.metric("TT Avg Rating",          f"{tt_avg_r:.2f} \u2605"  if tt_avg_r    is not None else "\u2014")
        c2.metric("Competitor Avg Rating",   f"{comp_avg_r:.2f} \u2605" if comp_avg_r  is not None else "\u2014")
        c3.metric("TT Avg Reviews",          f"{tt_avg_rev:,.0f}"      if tt_avg_rev  is not None else "\u2014")
        c4.metric("Competitor Avg Reviews",  f"{comp_avg_rev:,.0f}"    if comp_avg_rev is not None else "\u2014")
    else:
        st.info("No competitor listings in current filter.")

    st.divider()

    # ── Brand Defense Signal
    st.subheader("Brand Defense Signal")
    st.caption(
        "Listings claiming compatibility with Tommee Tippee products "
        "compete indirectly by targeting the same customers."
    )

    if "is_compatible_accessory" in df.columns:
        compat = df[df["is_compatible_accessory"]]
        n_compat = len(compat)
        n_compat_nottt = (
            len(compat[~compat["is_tommee_tippee_brand"]])
            if "is_tommee_tippee_brand" in compat.columns else n_compat
        )
        bd1, bd2 = st.columns(2)
        bd1.metric("Listings claiming TT compatibility", n_compat)
        bd2.metric(
            "Of which NOT by Tommee Tippee", n_compat_nottt,
            delta=f"{'⚠ brand intrusion' if n_compat_nottt > 5 else 'low risk'}",
            delta_color="inverse",
        )
        if n_compat > 0:
            with st.expander("See compatibility-claiming listings", expanded=False):
                compat_cols = [c for c in ["title", "brand", "price_value", "stars"] if c in compat.columns]
                compat_display = compat[compat_cols].copy()
                rename_map = {"title": "Title", "brand": "Brand", "price_value": "Price ($)", "stars": "Rating"}
                compat_display.columns = [rename_map.get(c, c) for c in compat_cols]
                st.dataframe(compat_display, use_container_width=True)
    else:
        st.info("Compatibility data not available.")

    st.divider()

    # ── Title Intelligence
    st.subheader("Title Intelligence")
    st.caption("Top keywords in Tommee Tippee vs competitor titles within this search space.")

    col_ti1, col_ti2 = st.columns(2)
    with col_ti1:
        st.markdown("**Tommee Tippee**")
        if not tt.empty and "title" in tt.columns:
            tt_kw = extract_keyword_frequencies(tt["title"].dropna(), TITLE_PHRASE_MAP, TITLE_STOPWORDS, top_n=15)
            if not tt_kw.empty:
                fig_tt_kw = px.bar(tt_kw, x="frequency", y="keyword", orientation="h",
                                   color_discrete_sequence=[TT_BLUE])
                fig_tt_kw.update_layout(yaxis={"autorange": "reversed"}, showlegend=False,
                                        xaxis_title="Frequency", yaxis_title="")
                st.plotly_chart(fig_tt_kw, use_container_width=True)
        else:
            st.info("No Tommee Tippee listings in current filter.")

    with col_ti2:
        st.markdown("**Competitors**")
        if not comp.empty and "title" in comp.columns:
            comp_kw = extract_keyword_frequencies(comp["title"].dropna(), TITLE_PHRASE_MAP, TITLE_STOPWORDS, top_n=15)
            if not comp_kw.empty:
                fig_comp_kw = px.bar(comp_kw, x="frequency", y="keyword", orientation="h",
                                     color_discrete_sequence=[COMP_ORANGE])
                fig_comp_kw.update_layout(yaxis={"autorange": "reversed"}, showlegend=False,
                                          xaxis_title="Frequency", yaxis_title="")
                st.plotly_chart(fig_comp_kw, use_container_width=True)
        else:
            st.info("No competitor listings in current filter.")

    if not tt.empty and not comp.empty and "title" in df.columns:
        st.markdown("**Title Attribute Coverage — TT vs Competitors**")
        cov_rows = []
        for component, pattern in TITLE_COMPONENTS.items():
            tt_hits   = tt["title"].dropna().str.contains(pattern, case=False, regex=True).sum()
            comp_hits = comp["title"].dropna().str.contains(pattern, case=False, regex=True).sum()
            cov_rows.append({
                "Attribute":       component,
                "Tommee Tippee %": round(tt_hits   / max(len(tt),   1) * 100, 1),
                "Competitors %":   round(comp_hits / max(len(comp), 1) * 100, 1),
            })
        cov_df = pd.DataFrame(cov_rows)
        fig_cov = px.bar(
            cov_df.melt(id_vars="Attribute", var_name="Brand Group", value_name="% of Titles"),
            x="Attribute", y="% of Titles", color="Brand Group", barmode="group",
            color_discrete_map={"Tommee Tippee %": TT_BLUE, "Competitors %": COMP_ORANGE},
        )
        fig_cov.update_layout(legend_title="")
        st.plotly_chart(fig_cov, use_container_width=True)

    st.divider()

    # ── Marketing Claims
    st.subheader("Marketing Claims")
    st.caption("Which product claims appear in descriptions — Tommee Tippee vs competitors.")

    if "description" in df.columns and not comp.empty and not tt.empty:
        def _count_claims_pct(subset: pd.DataFrame, n: int) -> pd.Series:
            desc_series = subset["description"].dropna().astype(str)
            counts: dict = {}
            for claim_label, pattern in CLAIM_MAP.items():
                hits = desc_series.str.contains(pattern, case=False, regex=True).sum()
                if hits > 0:
                    counts[claim_label] = round(hits / max(n, 1) * 100, 1)
            return pd.Series(counts)

        tt_c   = _count_claims_pct(tt,   len(tt)).rename("Tommee Tippee")
        comp_c = _count_claims_pct(comp, len(comp)).rename("Competitors")
        claims_df = pd.concat([tt_c, comp_c], axis=1).fillna(0)
        claims_df = claims_df[claims_df.sum(axis=1) > 0].sort_values("Tommee Tippee", ascending=False)

        if not claims_df.empty:
            claims_long = (
                claims_df.reset_index()
                .rename(columns={"index": "Claim"})
                .melt(id_vars="Claim", var_name="Group", value_name="% of listings")
            )
            fig_cl = px.bar(
                claims_long, x="% of listings", y="Claim", color="Group",
                barmode="group", orientation="h",
                color_discrete_map={"Tommee Tippee": TT_BLUE, "Competitors": COMP_ORANGE},
                title="Marketing Claim Usage — % of Listings",
            )
            fig_cl.update_layout(yaxis={"autorange": "reversed"}, legend_title="")
            st.plotly_chart(fig_cl, use_container_width=True)
        else:
            st.info("No claim patterns matched in descriptions.")
    elif "description" not in df.columns:
        st.info("No description data available.")

    st.divider()

    # ── Pricing Positioning
    st.subheader("Pricing Positioning")
    st.caption("How Tommee Tippee prices relative to competitors within this search space.")

    priced_df = df[df["price_value"].notna()] if "price_value" in df.columns else pd.DataFrame()
    if not priced_df.empty and "is_tommee_tippee_brand" in priced_df.columns:
        col_pp1, col_pp2 = st.columns(2)
        with col_pp1:
            fig_ph = px.histogram(
                priced_df, x="price_value",
                color="is_tommee_tippee_brand",
                color_discrete_map={True: TT_BLUE, False: "#aaaaaa"},
                barmode="overlay", nbins=30,
                labels={"price_value": "Price ($)", "is_tommee_tippee_brand": "TT Brand"},
                title="Price Distribution — TT (blue) vs Others (grey)",
            )
            fig_ph.update_layout(showlegend=False)
            st.plotly_chart(fig_ph, use_container_width=True)

        with col_pp2:
            price_summary = []
            for label, subset in [("Tommee Tippee", tt), ("Competitors", comp)]:
                if not subset.empty and "price_value" in subset.columns:
                    p = subset["price_value"].dropna()
                    if not p.empty:
                        price_summary.append({
                            "Brand Group": label,
                            "Min ($)":    round(p.min(), 2),
                            "Median ($)": round(p.median(), 2),
                            "Mean ($)":   round(p.mean(), 2),
                            "Max ($)":    round(p.max(), 2),
                            "Listings":   len(p),
                        })
            if price_summary:
                st.markdown("**Price Summary**")
                st.dataframe(pd.DataFrame(price_summary), use_container_width=True, hide_index=True)

            if "price_flag" in df.columns:
                st.markdown("**Price Tier Mix**")
                tier_order = ["budget", "mid", "premium", "luxury", "unknown"]
                tt_tiers   = tt["price_flag"].value_counts().reindex(tier_order, fill_value=0).rename("TT")   if not tt.empty   else pd.Series(dtype=int)
                comp_tiers = comp["price_flag"].value_counts().reindex(tier_order, fill_value=0).rename("Comp") if not comp.empty else pd.Series(dtype=int)
                tier_df = pd.concat([tt_tiers, comp_tiers], axis=1)
                tier_df.index.name = "Price Tier"
                st.dataframe(tier_df, use_container_width=True)
    else:
        st.info("No price data available for current filter.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: PRODUCT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

def tab_product_explorer(df: pd.DataFrame) -> None:
    st.header("Product Explorer")
    st.caption("Search and browse Amazon listings in the filtered dataset.")

    search_term = st.text_input(
        "Search by title or brand",
        placeholder="e.g. anti colic, Dr. Brown's, slow flow"
    )

    if search_term:
        mask = pd.Series(False, index=df.index)
        for col in ["title", "brand"]:
            if col in df.columns:
                mask |= df[col].astype(str).str.contains(search_term, case=False, na=False)
        results = df[mask].copy()
    else:
        results = df.copy()

    st.caption(f"**{len(results):,}** listings found.")

    if results.empty:
        st.info("No listings match your search.")
        return

    # Thumbnail preview
    img_col = "thumbnailImage" if "thumbnailImage" in results.columns else None
    if img_col:
        img_df = results[
            results[img_col].notna() & results[img_col].astype(str).str.startswith("http")
        ].copy()
        if "reviewsCount" in img_df.columns:
            img_df = img_df.sort_values("reviewsCount", ascending=False)
        if not img_df.empty:
            with st.expander("Thumbnail Preview (top 16)", expanded=False):
                render_image_grid(img_df.head(16).reset_index(drop=True), img_col)

    # Data table
    table_cols = [c for c in ["title", "brand", "stars", "reviewsCount", "price_value", "url"]
                  if c in results.columns]
    table_df = (
        results[table_cols]
        .sort_values("reviewsCount", ascending=False)
        .reset_index(drop=True)
    )
    col_cfg: dict = {}
    if "stars" in table_df.columns:
        col_cfg["stars"] = st.column_config.NumberColumn("Rating", format="%.1f \u2605")
    if "reviewsCount" in table_df.columns:
        col_cfg["reviewsCount"] = st.column_config.NumberColumn("Reviews", format="%d")
    if "price_value" in table_df.columns:
        col_cfg["price_value"] = st.column_config.NumberColumn("Price", format="$%.2f")
    if "url" in table_df.columns:
        col_cfg["url"] = st.column_config.LinkColumn("Link")
    st.dataframe(table_df, column_config=col_cfg, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.title("Tommee Tippee — Amazon Competitive Intelligence")
    st.caption(
        "A focused view of the Amazon baby bottle competitive landscape, "
        "designed to help Tommee Tippee understand its position and opportunities. "
        "Use the sidebar to filter by brand, price, or rating."
    )

    df_raw = load_data(CLEANED_FILE)
    if df_raw.empty:
        st.error(
            "Data file not found or empty. "
            "Run `scripts/clean_data.py` first to generate the dataset."
        )
        return

    df = apply_sidebar_filters(df_raw)
    if df.empty:
        st.warning(
            "No products match the current filters. "
            "Try widening your selection in the sidebar."
        )
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Market Overview",
        "Title Intelligence",
        "Messaging & Claims",
        "Branded Search",
        "Product Explorer",
    ])

    with tab1:
        tab_overview(df)
    with tab2:
        tab_title_intelligence(df)
    with tab3:
        tab_messaging(df)
    with tab4:
        tab_branded_search()
    with tab5:
        tab_product_explorer(df)


if __name__ == "__main__":
    main()
