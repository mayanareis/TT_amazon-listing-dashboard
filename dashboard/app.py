"""Amazon Competitive Listing Intelligence Dashboard.

Analyzes how competitors present their product listings on Amazon,
focusing on title strategy, description content, and visual presentation.

Organized into six tabs:
  Overview | Title Intelligence | Description Intelligence |
  Image Analysis | Product Explorer | Tommee Tippee Deep Dive
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
    page_title="Amazon Competitive Listing Intelligence",
    layout="wide",
    page_icon="🛒",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_FILE = PROJECT_ROOT / "data" / "cleaned" / "amazon_products_clean.csv"


# ══════════════════════════════════════════════════════════════════════════════
# PHRASE MAPS
# Each entry is (regex_pattern, underscore_token).
# ORDER MATTERS — longer / more specific phrases must come first so they are
# matched before their component words are consumed by shorter patterns.
# Underscore tokens produced here survive the stopword filter and are later
# rendered as "Title Case With Spaces" for chart labels.
# ══════════════════════════════════════════════════════════════════════════════

# Used for title analysis
TITLE_PHRASE_MAP: List[Tuple[str, str]] = [
    # ── Brand names must come FIRST so their component words are never split.
    # Without this, "Tommee Tippee" becomes two tokens "tommee" + "tippee" and
    # "Dr Brown's" becomes "dr" + "brown" — both meaningless in isolation.
    (r"\btommee\s+tippee\b",          "tommee_tippee"),
    (r"\bdr\.?\s+brown'?s?\b",        "dr_browns"),    # "Dr Brown's" / "Dr Browns"
    (r"\bdr\.?\s+brown\b",            "dr_brown"),     # fallback for bare "Dr Brown"
    # AirFree / Air Free (brand-specific, very specific → first)
    (r"\bairfree\s+vent\b",          "airfree_vent"),
    (r"\bairfree\b",                  "airfree_vent"),
    (r"\bair[-\s]+free\b",            "air_free"),
    # Natural Response / Natural Latch (Philips / general brand terms)
    (r"\bnatural\s+response\b",       "natural_response"),
    (r"\bnatural\s+bottle\b",         "natural_bottle"),
    (r"\bnatural\s+latch\b",          "natural_latch"),
    # Bottle types (specific first, generic last)
    (r"\bglass\s+bottle\b",           "glass_bottle"),
    (r"\bbaby\s+bottles?\b",          "baby_bottle"),
    # Part / material phrases
    (r"\bsilicone\s+nipple\b",        "silicone_nipple"),
    # Functional claims
    (r"\banti[-\s]+colic\b",          "anti_colic"),
    (r"\bbpa[-\s]+free\b",            "bpa_free"),
    (r"\bleak[-\s]+proof\b",          "leak_proof"),
    (r"\bdishwasher[-\s]+safe\b",     "dishwasher_safe"),
    (r"\bbreast[-\s]+like\b",         "breast_like"),
    # Ergonomic / handling phrases
    (r"\beasy\s+hold\b",              "easy_hold"),
    (r"\beasy\s+grip\b",              "easy_grip"),
    # Flow rates
    (r"\bslow\s+flow\b",              "slow_flow"),
    (r"\bmedium\s+flow\b",            "medium_flow"),
    (r"\bfast\s+flow\b",              "fast_flow"),
    (r"\bfirst\s+flow\b",             "first_flow"),
    # Physical attributes
    (r"\bwide\s+neck\b",              "wide_neck"),
    # Transition phrases
    (r"\bbreast\s+to\s+bottle\b",     "breast_to_bottle"),
    (r"\bbottle\s+to\s+breast\b",     "breast_to_bottle"),
    # Cleaning / care
    (r"\bself[-\s]+steriliz\w*\b",    "self_sterilizing"),
    # Developmental
    (r"\bseamless\s+transition\b",    "seamless_transition"),
    # Sippy / training (captured so they don't split — excluded downstream)
    (r"\bsippy\s+cup\b",              "sippy_cup"),
    (r"\bsip\s+cup\b",                "sippy_cup"),
]

# Used for description analysis — superset of title phrases plus
# description-specific patterns (feeding context, cleaning claims, etc.)
DESC_PHRASE_MAP: List[Tuple[str, str]] = [
    # ── Brand names first — same reasoning as TITLE_PHRASE_MAP above.
    (r"\btommee\s+tippee\b",          "tommee_tippee"),
    (r"\bdr\.?\s+brown'?s?\b",        "dr_browns"),
    (r"\bdr\.?\s+brown\b",            "dr_brown"),
    # AirFree / Air Free
    (r"\bairfree\s+vent\b",           "airfree_vent"),
    (r"\bairfree\b",                   "airfree_vent"),
    (r"\bair[-\s]+free\b",             "air_free"),
    # Natural terms
    (r"\bnatural\s+response\b",        "natural_response"),
    (r"\bnatural\s+latch\b",           "natural_latch"),
    # Feeding context phrases (description-specific)
    (r"\bbottle[-\s]+feeding\b",       "bottle_feeding"),
    (r"\bfeeding\s+bottle\b",          "feeding_bottle"),
    (r"\bbreast\s*feeding\b",          "breastfeeding"),
    (r"\bbreast[-\s]+like\b",          "breast_like"),
    (r"\bbreast\s+feed\b",             "breastfeed"),
    # Bottle types
    (r"\bglass\s+bottle\b",            "glass_bottle"),
    (r"\bbaby\s+bottles?\b",           "baby_bottle"),
    # Part / material
    (r"\bsilicone\s+nipple\b",         "silicone_nipple"),
    (r"\bsoft\s+silicone\b",           "soft_silicone"),
    # Functional claims
    (r"\banti[-\s]+colic\b",           "anti_colic"),
    (r"\bbpa[-\s]+free\b",             "bpa_free"),
    (r"\bleak[-\s]+proof\b",           "leak_proof"),
    (r"\bdishwasher[-\s]+safe\b",      "dishwasher_safe"),
    # Cleaning claims
    (r"\beasy[-\s]+to[-\s]+clean\b",   "easy_clean"),
    (r"\beasy[-\s]+clean\b",           "easy_clean"),
    # Ergonomic / handling phrases
    (r"\beasy\s+hold\b",               "easy_hold"),
    (r"\beasy\s+grip\b",               "easy_grip"),
    # Flow rates
    (r"\bslow\s+flow\b",               "slow_flow"),
    (r"\bmedium\s+flow\b",             "medium_flow"),
    (r"\bfast\s+flow\b",               "fast_flow"),
    (r"\bfirst\s+flow\b",              "first_flow"),
    # Physical attributes
    (r"\bwide\s+neck\b",               "wide_neck"),
    # Feeding position
    (r"\bupright\s+feeding\b",         "upright_feeding"),
    # Transition phrases
    (r"\bbreast\s+to\s+bottle\b",      "breast_to_bottle"),
    (r"\bbottle\s+to\s+breast\b",      "breast_to_bottle"),
    # Cleaning / care
    (r"\bself[-\s]+steriliz\w*\b",     "self_sterilizing"),
    # Developmental
    (r"\bseamless\s+transition\b",     "seamless_transition"),
    # Colic / vent system
    (r"\bvent\s+system\b",             "vent_system"),
    (r"\banti[-\s]?colic\s+system\b",  "anti_colic_system"),
    # Material-led phrases
    (r"\bsoft\s+nipple\b",             "soft_nipple"),
    (r"\bnatural\s+nipple\b",          "natural_nipple"),
    # Sippy / training (captured so they don't split — excluded downstream)
    (r"\bsippy\s+cup\b",               "sippy_cup"),
    (r"\bsip\s+cup\b",                 "sippy_cup"),
]


# ══════════════════════════════════════════════════════════════════════════════
# STOPWORD SETS
# Applied only to plain single-word tokens. Underscore phrase tokens always
# pass through regardless — they are meaningful by construction.
# ══════════════════════════════════════════════════════════════════════════════

# Title stopwords: remove structural/grammatical noise AND category-generic
# nouns that would otherwise dominate every baby-product chart.
TITLE_STOPWORDS: set = {
    # Common English
    "the", "and", "for", "with", "this", "that", "from", "your", "you",
    "are", "was", "were", "have", "has", "had", "will", "can", "our",
    "its", "all", "not", "but", "use", "using", "new", "amazon",
    "in", "on", "of", "to", "a", "an", "is", "it", "be",
    "or", "by", "at", "as", "up", "do", "if", "so", "no", "we", "my",
    "he", "she", "they", "them", "their", "also", "into", "than", "other",
    "each", "more", "made", "make", "when", "which", "who", "what", "how",
    "get", "give", "just", "out", "any", "one", "two", "may", "well",
    # Category-generic nouns — too noisy for baby bottle titles
    "baby", "bottle", "bottles",
    # Units & sizing noise
    "oz", "ml", "pk", "count",
    # Age/time noise
    "month", "months",
    # Quantity noise
    "pack", "set", "piece", "pieces", "product",
    # Brand-name fragments — ALL individual brand words must be here so they
    # never surface as keywords. Compound brand tokens (e.g. tommee_tippee) are
    # handled separately via BRAND_TOKENS below.
    "tommee", "tippee", "brown", "dr",
    "philips", "avent", "nuk", "evenflo", "mam",
    "comotomo", "nanobebe", "lansinoh", "medela",
    "chicco", "playtex", "gerber", "first", "years",
}

# Description stopwords: heavier filtering for the longer prose found in
# product descriptions, where structural verbs dominate otherwise.
DESC_STOPWORDS: set = {
    # Common English
    "the", "and", "for", "with", "that", "this", "your", "from", "into",
    "only", "are", "has", "have", "had", "will", "can", "our", "its",
    "all", "not", "but", "use", "used", "using", "new", "amazon",
    "in", "on", "of", "to", "a", "an", "is", "it", "be", "or", "by",
    "at", "as", "up", "do", "if", "so", "no", "we", "my", "you", "was",
    "were", "also", "each", "more", "when", "which", "who", "what", "how",
    "get", "just", "any", "one", "two", "may", "well", "than", "other",
    "made", "make", "them", "they", "their", "while", "both",
    # Description-specific filler verbs / passive constructions
    "designed", "includes", "included", "featuring", "features", "help",
    "helps", "helping", "keep", "keeps", "allow", "allows", "give", "gives",
    "reduce", "reduces", "providing", "provides", "ensure", "ensures",
    "prevent", "prevents", "support", "supports", "comes", "come",
    # Category-generic nouns
    "baby", "bottle", "bottles", "feeding", "product", "products",
    "nipple", "nipples",   # too common alone; the phrase "silicone nipple" is kept via phrase map
    # Units
    "oz", "ml",
    # Brand-name fragments — same logic as TITLE_STOPWORDS
    "tommee", "tippee", "brown", "dr",
    "philips", "avent", "nuk", "evenflo", "mam",
    "comotomo", "nanobebe", "lansinoh", "medela",
    "chicco", "playtex", "gerber", "first", "years",
}


# ══════════════════════════════════════════════════════════════════════════════
# CLAIM MAP  (for listing-level presence detection)
# Each key is a human-readable label; value is the regex to detect it.
# Counts are at the LISTING level: if a description says "anti-colic" 5×,
# it still counts as 1 listing containing that claim.
# ══════════════════════════════════════════════════════════════════════════════
CLAIM_MAP: dict = {
    "Anti-Colic":        r"\banti[-\s]?colic\b",
    "BPA Free":          r"\bbpa[-\s]?free\b",
    "Natural Response":  r"\bnatural\s+response\b",
    "Leak Proof":        r"\bleak[-\s]?proof\b",
    "Dishwasher Safe":   r"\bdishwasher[-\s]?safe\b",
    "Easy Clean":        r"\beasy[-\s]?(?:to[-\s]?)?clean\b",
    "Breast-Like":       r"\bbreast[-\s]?like\b",
    "Slow Flow":         r"\bslow\s+flow\b",
    "Medium Flow":       r"\bmedium\s+flow\b",
    "Fast Flow":         r"\bfast\s+flow\b",
    "Silicone Nipple":   r"\bsilicone\s+nipple\b",
    "Wide Neck":         r"\bwide\s+neck\b",
}

# Display-label overrides for underscore tokens where plain title-case is wrong.
# token_to_label() applies these before falling back to .replace("_"," ").title().
LABEL_OVERRIDES: dict = {
    # Acronyms / special capitalisation
    "bpa_free":       "BPA Free",
    "airfree_vent":   "AirFree Vent",
    "air_free":       "Air Free",
    # Brand names — title-case produces the right result but we pin them
    # explicitly so the intent is clear and easy to extend.
    "tommee_tippee":  "Tommee Tippee",
    "dr_browns":      "Dr Brown's",
    "dr_brown":       "Dr Brown",
}


# Compound brand tokens produced by the phrase maps that must NEVER appear
# as keywords in any frequency chart. The phrase maps still need these patterns
# to prevent brand words from splitting into meaningless fragments — but the
# resulting tokens are blocked here before they reach the output.
BRAND_TOKENS: set = {
    # Brand compound tokens — must never appear as keywords
    "tommee_tippee", "dr_browns", "dr_brown",
    "philips_avent", "evenflo", "first_years",
    "mam", "nuk", "comotomo", "nanobebe",
    "lansinoh", "medela", "chicco", "playtex", "gerber",
    # Product-type tokens for excluded categories (sippy cups etc.)
    # Matched in phrase map to avoid splitting but not shown as keywords
    "sippy_cup",
}


# ══════════════════════════════════════════════════════════════════════════════
# BRAND NORMALIZATION MAP
# Applied once at load time to canonicalize raw brand strings.
# Prevents compound brand names (e.g. "Dr. Brown's" / "Dr Browns") from
# appearing as separate entries in charts and filters.
# ══════════════════════════════════════════════════════════════════════════════
BRAND_NORMALIZATION_MAP: List[Tuple[str, str]] = [
    # (case-insensitive regex, canonical brand name)  — most specific first
    (r"tommee\s+tippee",        "Tommee Tippee"),
    (r"dr\.?\s+brown'?s?",      "Dr. Brown's"),
    (r"philips\s+avent",        "Philips Avent"),
    (r"evenflo\s+feeding",      "Evenflo"),
    (r"the\s+first\s+years?",   "The First Years"),
    (r"\bfirst\s+years?\b",     "The First Years"),
    (r"\bmam\b",                "MAM"),
    (r"\bnuk\b",                "NUK"),
    (r"comotomo",               "Comotomo"),
    (r"nanobebe",               "Nanobebe"),
    (r"lansinoh",               "Lansinoh"),
    (r"medela",                 "Medela"),
    (r"chicco",                 "Chicco"),
    (r"playtex",                "Playtex"),
    (r"gerber",                 "Gerber"),
]


def normalize_brand(raw_brand: str) -> str:
    """Return a canonical brand name from a raw brand string.

    Applies BRAND_NORMALIZATION_MAP to consolidate variants like
    'Dr. Brown's' / 'Dr Browns' / 'dr brown' → 'Dr. Brown's'.
    Falls back to title-casing the raw value if no pattern matches.
    """
    if not isinstance(raw_brand, str) or not raw_brand.strip():
        return "Unknown"
    b = raw_brand.strip()
    for pattern, canonical in BRAND_NORMALIZATION_MAP:
        if re.search(pattern, b, flags=re.IGNORECASE):
            return canonical
    return b.title() if b else "Unknown"


# ══════════════════════════════════════════════════════════════════════════════
# TITLE COMPONENT PATTERNS
# Used to detect how many structural elements a title contains.
# Title Quality Score = (components detected / total components) × 100.
# ══════════════════════════════════════════════════════════════════════════════
# Title Quality Score components — tuned specifically for baby bottles and
# bottle nipples. Each component is weighted equally (1 point each).
# Brand names are intentionally absent — scores reflect product language only,
# not how frequently a brand appears in the dataset.
TITLE_COMPONENTS: dict = {
    # Core product type — bottle or nipple only (sippy cups excluded)
    "Product Type":  r"\b(?:bottle|bottles?|nipple|nipples?)\b",
    # Flow rate — primary differentiator for nipples
    "Flow Rate":     r"\b(?:slow\s+flow|medium\s+flow|fast\s+flow|first\s+flow|variable\s+flow|newborn\s+flow)\b",
    # Anti-colic — a key purchase driver in this category
    "Anti-Colic":    r"\b(?:anti[-\s]?colic|vented?|vent\s+system|airfree|air[-\s]free)\b",
    # Size or volume (oz, ml)
    "Size / Volume": r"\b\d+\s*(?:oz|ml|fl\.?\s*oz)\b",
    # Pack quantity — value signal
    "Quantity":      r"\b(?:\d+\s*[-–]?\s*(?:pack|count|ct\b)|pack\s+of\s+\d+|set\s+of\s+\d+|\d+\s+(?:bottles?|nipples?))\b",
    # Material or safety claim
    "Material":      r"\b(?:glass|silicone|stainless|steel|bpa[-\s]?free|ppsu|tritan)\b",
    # Compatibility claim
    "Compatibility": r"\b(?:compatible\s+with|works\s+with|fits\s+(?:most|all)?|universal)\b",
}


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE vs BENEFIT LANGUAGE PATTERNS
# Feature language = what the product has/is made of (specs-led).
# Benefit language = what the product does for the customer (outcome-led).
# Used to estimate each listing's messaging orientation.
# ══════════════════════════════════════════════════════════════════════════════
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
    r"\b\d+\s*[-–]?\s*(?:pack|count)\b",
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
# DESCRIPTION SECTION PATTERNS
# Detect whether descriptions include key marketing copy sections.
# Description Quality Score = sections covered / total sections × 100.
# ══════════════════════════════════════════════════════════════════════════════
DESC_SECTION_PATTERNS: dict = {
    "Opening Claim":    [
        r"\bdesigned\s+(?:for|to)\b", r"\bperfect\s+for\b", r"\bideal\s+for\b",
        r"\bintroducing\b", r"\bthe\s+(?:best|ultimate|only)\b",
    ],
    "Feature Details":  [
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
    "Age & Fit":        [
        r"\b(?:newborn|infant|0\+?\s*m(?:onths?)?|\d+\+?\s*m(?:onths?)?)\b",
        r"\bsuitable\s+for\b", r"\bcompatible\s+with\b", r"\bworks\s+with\b",
    ],
    "Care & Usage":     [
        r"\bdishwasher[-\s]?safe\b", r"\beasy[-\s](?:to[-\s])?clean\b",
        r"\bsteriliz\w+\b", r"\bhow\s+to\s+use\b", r"\bwash\s+before\b",
    ],
    "Trust Signals":    [
        r"(?:#\s*1|no\.?\s*1|number\s+one|no\s+1)\s+brand",
        r"\brecommended\s+by\b", r"\bclinically\b",
        r"\baward\b", r"\btrusted\b", r"\bmillion\w*\s+(?:of\s+)?moms?\b",
        r"\bpediatrician\b",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# MESSAGING CATEGORIES
# Used by the Description Messaging Structure analysis.
# Each category maps to a list of regex patterns; a listing "covers" the
# category if ANY pattern matches its description (binary, not frequency).
# ══════════════════════════════════════════════════════════════════════════════
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

# Client-facing example phrases shown alongside each messaging category
# in the dashboard. These help clients understand what each category means
# without needing to see the underlying regex patterns.
MESSAGING_CATEGORY_EXAMPLES: dict = {
    "Social Proof": [
        "#1 brand recommended by moms",
        "trusted by millions of parents",
        "award-winning design",
        "recommended by pediatricians",
    ],
    "Safety Claims": [
        "BPA-free materials",
        "FDA-approved",
        "food-grade silicone",
        "non-toxic and hypoallergenic",
    ],
    "Comfort Claims": [
        "reduces colic and gas",
        "breast-like feel for natural latch",
        "anti-colic vent system",
        "gentle on baby's tummy",
    ],
    "Convenience Claims": [
        "dishwasher safe — top rack",
        "easy to clean, just 4 parts",
        "leak-proof design",
        "travel-friendly",
    ],
    "Design Claims": [
        "wide neck for easy filling",
        "soft silicone nipple",
        "ergonomic grip",
        "compact and lightweight",
    ],
    "Research / Survey Claims": [
        "clinically tested to reduce colic",
        "recommended by pediatricians",
        "proven by survey of 300 moms",
        "scientifically designed",
    ],
}


# Specific social proof phrases for fine-grained detection within the
# "Social Proof" category above.
SOCIAL_PROOF_PATTERNS: dict = {
    "#1 Brand":                 r"(?:#\s*1|no\.?\s*1|number\s+one|no\s+1)\s+brand",
    "Recommended by Moms":      r"(?:recommended\s+by\s+moms?|moms?\s+recommend)",
    "Recommended by Doctors":   r"(?:recommended\s+by\s+doctors?|doctor[s]?\s+recommended|doctors?\s+recommend)",
    "Clinically Tested/Proven": r"\bclinically\s+(?:tested|proven|validated|studied)\b",
    "Survey / Study":           r"\bsurvey\b|\bstud(?:y|ied)\b",
    "Award Winning":            r"\baward[- ]?winning\b",
    "Trusted By":               r"\btrusted\s+by\b",
    "Millions of Moms":         r"(?:million[s]?\s+(?:of\s+)?moms?|trusted\s+by\s+million[s]?)",
    "Pediatrician Recommended": r"(?:pediatrician[s]?\s*(?:recommended|approved|tested)?|recommended\s+by\s+pediatricians?)",
    "Trusted by Parents":       r"\btrusted\s+by\s+(?:parents?|families|caregivers?)\b",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def token_to_label(token: str) -> str:
    """Convert an internal underscore token to a human-readable chart label.

    Examples:
        "anti_colic"   -> "Anti Colic"
        "bpa_free"     -> "BPA Free"   (via LABEL_OVERRIDES)
        "airfree_vent" -> "AirFree Vent"
        "nipple"       -> "Nipple"
    """
    if token in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[token]
    return token.replace("_", " ").title()


def normalize_text_for_phrases(text: str, phrase_map: List[Tuple[str, str]]) -> str:
    """Replace known multi-word / hyphenated phrases with underscore tokens.

    Must run BEFORE word tokenization so that e.g. "anti-colic" becomes the
    single token "anti_colic" rather than two separate words "anti" and "colic".

    Args:
        text:       Raw input string (will be lowercased internally).
        phrase_map: Ordered list of (regex_pattern, replacement_token) pairs.
                    More-specific patterns must appear before shorter ones.

    Returns:
        Lowercased string with recognised phrases replaced by underscore tokens.
    """
    # Lowercase once so all patterns can assume lowercase input
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
    """Normalize phrases, tokenize, filter stopwords, return top-N frequencies.

    Pipeline per document:
      1. Normalize: multi-word/hyphenated phrases → underscore tokens
      2. Tokenize: regex captures both plain words and underscore-joined tokens
      3. Filter:   stopwords removed for plain words; phrase tokens always kept
      4. Count:    aggregate with Counter across all documents
      5. Label:    convert tokens back to readable display names

    Args:
        series:     pandas Series of text strings (titles or descriptions).
        phrase_map: Ordered phrase-replacement list (see TITLE_PHRASE_MAP etc.).
        stopwords:  Set of lowercase single words to exclude.
        top_n:      Number of top entries to return.

    Returns:
        DataFrame with columns ["keyword", "frequency"] (display labels, not tokens).
    """
    all_tokens: List[str] = []

    for text in series.dropna().astype(str):
        # Step 1 — replace multi-word phrases with underscore tokens
        normalized = normalize_text_for_phrases(text, phrase_map)

        # Step 2 — tokenize
        # Regex greedily matches underscore-joined tokens first (e.g. anti_colic),
        # then falls back to plain words. This preserves phrase tokens intact.
        raw_tokens = re.findall(r"[a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)+|[a-z][a-z0-9]*", normalized)

        for tok in raw_tokens:
            if "_" in tok:
                # Step 3a — phrase token: keep unless it is a brand name.
                # Brand compound tokens (e.g. tommee_tippee) are needed in the
                # phrase map to prevent word-splitting, but should never surface
                # as keywords — they reflect brand frequency, not product language.
                if tok not in BRAND_TOKENS:
                    all_tokens.append(tok)
            elif tok not in stopwords and len(tok) > 2:
                # Step 3b — plain word: apply stopword + length filter
                all_tokens.append(tok)

    # Step 4 — count
    freq_pairs = Counter(all_tokens).most_common(top_n)

    # Step 5 — convert tokens to display labels
    rows = [(token_to_label(tok), cnt) for tok, cnt in freq_pairs]
    return pd.DataFrame(rows, columns=["keyword", "frequency"])


def build_claim_presence(
    df: pd.DataFrame,
    text_col: str,
    claim_map: dict,
) -> pd.DataFrame:
    """Count how many distinct listings contain each marketing claim.

    Presence is binary per listing: a description mentioning "anti-colic"
    four times still contributes only 1 to that claim's count.

    Args:
        df:        Full (filtered) DataFrame.
        text_col:  Column containing the text to scan (e.g. "description").
        claim_map: Dict mapping human-readable claim label → regex pattern.

    Returns:
        DataFrame with columns ["claim", "listings", "% of Products"],
        sorted by listings descending.
    """
    text_series = df[text_col].fillna("").astype(str)
    total = len(df)
    results = []

    for label, pattern in claim_map.items():
        # .str.contains returns a boolean Series; .sum() counts True values
        count = int(text_series.str.contains(pattern, flags=re.IGNORECASE, regex=True).sum())
        pct = round(count / total * 100, 1) if total > 0 else 0.0
        results.append({"claim": label, "listings": count, "% of Products": f"{pct}%"})

    return (
        pd.DataFrame(results)
        .sort_values("listings", ascending=False)
        .reset_index(drop=True)
    )


# ── Gap analysis helpers (used by the Tommee Tippee Deep Dive tab) ─────────────

def build_keyword_gap_table(
    tt_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    text_col: str,
    phrase_map: List[Tuple[str, str]],
    stopwords: set,
    top_n: int = 40,
) -> pd.DataFrame:
    """Compare keyword frequency between Tommee Tippee and competitor products.

    Frequencies are normalised to "% of products in that group" so that
    the two groups (which have very different sizes) can be compared fairly.

    Args:
        tt_df:      Tommee Tippee listings subset.
        comp_df:    All other competitor listings.
        text_col:   Column to analyse ("title", "clean_title", or "description").
        phrase_map: Phrase map to use (TITLE_PHRASE_MAP or DESC_PHRASE_MAP).
        stopwords:  Stopword set to use (TITLE_STOPWORDS or DESC_STOPWORDS).
        top_n:      Number of top keywords to extract per group before merging.

    Returns:
        DataFrame with columns:
          Keyword / Phrase | TT Count | TT % | Comp Count | Comp % | Gap % (Comp − TT)
        Sorted by Gap % descending (biggest competitor advantage first).
    """
    tt_kw   = extract_keyword_frequencies(tt_df[text_col].dropna(),   phrase_map, stopwords, top_n=top_n)
    comp_kw = extract_keyword_frequencies(comp_df[text_col].dropna(), phrase_map, stopwords, top_n=top_n)

    tt_n   = max(len(tt_df), 1)
    comp_n = max(len(comp_df), 1)

    # Normalise raw counts → % of products in each group
    tt_kw   = tt_kw.copy();   tt_kw["TT %"]   = (tt_kw["frequency"]   / tt_n   * 100).round(1)
    comp_kw = comp_kw.copy(); comp_kw["Comp %"] = (comp_kw["frequency"] / comp_n * 100).round(1)

    merged = pd.merge(
        tt_kw.rename(columns={"frequency": "TT Count"})[["keyword", "TT Count", "TT %"]],
        comp_kw.rename(columns={"frequency": "Comp Count"})[["keyword", "Comp Count", "Comp %"]],
        on="keyword",
        how="outer",
    ).fillna(0)

    merged["TT Count"]   = merged["TT Count"].astype(int)
    merged["Comp Count"] = merged["Comp Count"].astype(int)

    # Positive gap = competitors use this keyword more → opportunity for Tommee Tippee
    merged["Gap % (Comp \u2212 TT)"] = (merged["Comp %"] - merged["TT %"]).round(1)

    return (
        merged
        .rename(columns={"keyword": "Keyword / Phrase"})
        [["Keyword / Phrase", "TT Count", "TT %", "Comp Count", "Comp %", "Gap % (Comp \u2212 TT)"]]
        .sort_values("Gap % (Comp \u2212 TT)", ascending=False)
        .reset_index(drop=True)
    )


def build_claim_gap_table(
    tt_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    text_col: str,
    claim_map: dict,
) -> pd.DataFrame:
    """Compare marketing claim presence (% of products) between TT and competitors.

    Uses listing-level binary detection (same logic as build_claim_presence).

    Returns:
        DataFrame with columns:
          claim | TT Listings | TT % | Comp Listings | Comp % | Gap % (Comp − TT)
        Sorted by Gap % descending.
    """
    tt_n   = max(len(tt_df), 1)
    comp_n = max(len(comp_df), 1)

    # Extract raw listing counts from both groups
    tt_raw   = build_claim_presence(tt_df,   text_col, claim_map)[["claim", "listings"]].rename(columns={"listings": "TT Listings"})
    comp_raw = build_claim_presence(comp_df, text_col, claim_map)[["claim", "listings"]].rename(columns={"listings": "Comp Listings"})

    merged = pd.merge(tt_raw, comp_raw, on="claim", how="outer").fillna(0)
    merged["TT Listings"]   = merged["TT Listings"].astype(int)
    merged["Comp Listings"] = merged["Comp Listings"].astype(int)

    # Compute percentage per group independently
    merged["TT %"]   = (merged["TT Listings"]   / tt_n   * 100).round(1)
    merged["Comp %"] = (merged["Comp Listings"] / comp_n * 100).round(1)

    # Positive gap = competitor uses this claim more → opportunity for Tommee Tippee
    merged["Gap % (Comp \u2212 TT)"] = (merged["Comp %"] - merged["TT %"]).round(1)

    return (
        merged[["claim", "TT Listings", "TT %", "Comp Listings", "Comp %", "Gap % (Comp \u2212 TT)"]]
        .sort_values("Gap % (Comp \u2212 TT)", ascending=False)
        .reset_index(drop=True)
    )


def generate_tt_recommendations(
    tt_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    title_gap_df: pd.DataFrame,
    desc_gap_df: pd.DataFrame,
    claim_gap_df: pd.DataFrame,
    gap_threshold: float = 5.0,
) -> List[str]:
    """Generate rule-based text insights from gap analysis results.

    Each insight is a markdown-formatted string ready to render with st.markdown.

    Args:
        tt_df, comp_df:   Tommee Tippee and competitor DataFrames.
        title_gap_df:     Output of build_keyword_gap_table on titles.
        desc_gap_df:      Output of build_keyword_gap_table on descriptions.
        claim_gap_df:     Output of build_claim_gap_table.
        gap_threshold:    Minimum percentage-point gap to flag as an insight.

    Returns:
        List of markdown bullet strings.
    """
    gap_col = "Gap % (Comp \u2212 TT)"
    insights: List[str] = []

    # ── Title length comparison
    if "title_word_count" in tt_df.columns and len(tt_df) > 0 and len(comp_df) > 0:
        tt_avg   = tt_df["title_word_count"].mean()
        comp_avg = comp_df["title_word_count"].mean()
        diff = tt_avg - comp_avg
        if abs(diff) >= 1.0:
            direction = "shorter" if diff < 0 else "longer"
            action = (
                "Longer titles can incorporate more keywords, flow rates, and material callouts."
                if diff < 0
                else "Good length — ensure every word earns its place with a specific benefit."
            )
            insights.append(
                f"Tommee Tippee titles are on average **{abs(diff):.1f} words {direction}** "
                f"than competitors ({tt_avg:.1f} vs {comp_avg:.1f} words). {action}"
            )

    # ── Description length comparison
    if "description_length" in tt_df.columns and len(tt_df) > 0 and len(comp_df) > 0:
        tt_avg   = tt_df["description_length"].mean()
        comp_avg = comp_df["description_length"].mean()
        diff = tt_avg - comp_avg
        if abs(diff) >= 100:
            direction = "shorter" if diff < 0 else "longer"
            action = (
                "There may be an opportunity to expand description content with more benefit-led copy."
                if diff < 0
                else "Good content depth — ensure it leads with consumer benefits, not just features."
            )
            insights.append(
                f"Tommee Tippee descriptions are on average **{abs(diff):,.0f} characters {direction}** "
                f"than competitors ({tt_avg:,.0f} vs {comp_avg:,.0f} chars). {action}"
            )

    # ── Title keyword gaps (top 3 opportunities)
    title_opps = title_gap_df[title_gap_df[gap_col] > gap_threshold].head(3)
    for _, row in title_opps.iterrows():
        insights.append(
            f"Tommee Tippee titles underuse **{row['Keyword / Phrase']}** — competitors include it "
            f"in **{row['Comp %']:.1f}%** of listings vs **{row['TT %']:.1f}%** for Tommee Tippee. "
            f"Adding this phrase to titles could improve keyword visibility."
        )

    # ── Description theme gaps (top 3 opportunities)
    desc_opps = desc_gap_df[desc_gap_df[gap_col] > gap_threshold].head(3)
    for _, row in desc_opps.iterrows():
        insights.append(
            f"Tommee Tippee may have an opportunity to strengthen messaging around "
            f"**{row['Keyword / Phrase']}** — competitors use it in descriptions "
            f"**{row['Comp %']:.1f}%** of the time vs **{row['TT %']:.1f}%** for Tommee Tippee."
        )

    # ── Claim gaps (top 3 opportunities)
    claim_opps = claim_gap_df[claim_gap_df[gap_col] > gap_threshold].head(3)
    for _, row in claim_opps.iterrows():
        insights.append(
            f"Competitors more often include the **{row['claim']}** claim "
            f"(**{row['Comp %']:.1f}%** of listings vs **{row['TT %']:.1f}%** for Tommee Tippee). "
            f"If applicable, Tommee Tippee should make this claim more explicit."
        )

    if not insights:
        insights.append(
            "No significant content gaps detected with the current data and filters. "
            "Tommee Tippee appears well-aligned with competitor content patterns."
        )

    return insights


def build_messaging_category_scores(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Score each listing across the six messaging categories.

    Each category column is binary (1 = at least one pattern matched, 0 = none).
    claim_coverage_score = (categories present / total categories) × 100.

    Args:
        df:       DataFrame containing the text column.
        text_col: Column to analyse, e.g. "description".

    Returns:
        DataFrame (same index as df) with one bool-int column per category,
        a numeric 'claim_coverage_score' column, and 'brand' if available.
    """
    text_series = df[text_col].fillna("").astype(str)
    result = pd.DataFrame(index=df.index)
    n_categories = len(MESSAGING_CATEGORIES)

    for category, patterns in MESSAGING_CATEGORIES.items():
        # Any match within the category counts as covered (binary)
        combined = "|".join(f"(?:{p})" for p in patterns)
        result[category] = (
            text_series.str.contains(combined, flags=re.IGNORECASE, regex=True).astype(int)
        )

    # Coverage score: fraction of categories covered expressed as 0–100
    result["claim_coverage_score"] = (
        result[list(MESSAGING_CATEGORIES.keys())].sum(axis=1) / n_categories * 100
    ).round(1)

    if "brand" in df.columns:
        result["brand"] = df["brand"].values

    return result


def build_messaging_comparison(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    text_col: str,
    label_a: str = "Group A",
    label_b: str = "Group B",
) -> pd.DataFrame:
    """Compare messaging category presence rates (% of listings) between two groups.

    Args:
        df_a, df_b: Two DataFrames to compare (e.g. Tommee Tippee vs competitors).
        text_col:   Column to analyse.
        label_a/b:  Display labels for the two groups.

    Returns:
        Long-format DataFrame with columns [Category, Group, % of Listings],
        suitable for px.bar(barmode='group').
    """
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
    img_df: pd.DataFrame,
    img_col: str,
    images_per_row: int = 4,
) -> None:
    """Render a simple grid of product thumbnail images with brand + title captions.

    Used by the Tommee Tippee Deep Dive tab to avoid duplicating grid logic.

    Args:
        img_df:         DataFrame containing valid image rows.
        img_col:        Column name holding the image URL.
        images_per_row: Number of columns in each grid row.
    """
    for row_start in range(0, len(img_df), images_per_row):
        row_slice = img_df.iloc[row_start: row_start + images_per_row]
        cols = st.columns(images_per_row)
        for col, (_, product) in zip(cols, row_slice.iterrows()):
            with col:
                try:
                    st.image(str(product[img_col]), use_container_width=True)
                except Exception:
                    st.write("Image unavailable")
                brand_label = (
                    str(product.get("brand", "")) if pd.notna(product.get("brand")) else ""
                )
                raw_title = str(product.get("title", ""))
                title_label = raw_title[:55] + "\u2026" if len(raw_title) > 55 else raw_title
                st.caption(f"**{brand_label}**  \n{title_label}")


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCT CATEGORY FILTER
# ══════════════════════════════════════════════════════════════════════════════

def filter_bottles_and_nipples(df: pd.DataFrame) -> pd.DataFrame:
    """Return only baby bottle and bottle nipple listings.

    All keyword, phrase, scoring, and messaging analysis should run on this
    filtered subset to prevent sippy cups, warmers, sterilizers, and other
    accessories from skewing insights intended for the core bottle/nipple category.

    Inclusion rule: title or breadcrumbs contains 'bottle' or 'nipple'.
    Exclusion rule: title or breadcrumbs explicitly marks the product as a
    sippy cup, straw cup, warmer, sterilizer, or pacifier.
    """
    if df.empty:
        return df

    title_lc = df.get("title", pd.Series(dtype=str)).fillna("").str.lower()
    bc_lc     = df.get("breadcrumbs", pd.Series(dtype=str)).fillna("").str.lower()

    is_target = (
        title_lc.str.contains(r"\b(?:bottle|nipple)\b", regex=True)
        | bc_lc.str.contains(r"\b(?:bottle|nipple)\b", regex=True)
    )
    is_excluded = (
        title_lc.str.contains(r"\b(?:sippy|straw\s+cup|warmer|steriliz|pacifier|trainer\s+cup)\b", regex=True)
        | bc_lc.str.contains(r"\b(?:sippy|warmer|steriliz|pacifier)\b", regex=True)
    )

    return df[is_target & ~is_excluded].copy()


# ══════════════════════════════════════════════════════════════════════════════
# TITLE & DESCRIPTION QUALITY SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_title_quality_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Score each product title across TITLE_COMPONENTS.

    Returns a DataFrame with one binary column per component, a
    title_quality_score (0–100), and brand if available.
    Score logic: each detected component contributes equal weight.
    """
    title_col = "title" if "title" in df.columns else None
    if title_col is None or df.empty:
        return pd.DataFrame()

    text = df[title_col].fillna("").astype(str)
    result = pd.DataFrame(index=df.index)

    for component, pattern in TITLE_COMPONENTS.items():
        result[component] = text.str.contains(
            pattern, flags=re.IGNORECASE, regex=True
        ).astype(int)

    n_components = len(TITLE_COMPONENTS)
    result["title_quality_score"] = (
        result[list(TITLE_COMPONENTS.keys())].sum(axis=1) / n_components * 100
    ).round(1)

    if "brand" in df.columns:
        result["brand"] = df["brand"].values

    return result


def compute_description_quality_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Score each description across DESC_SECTION_PATTERNS.

    Returns a DataFrame with one binary column per section, a
    description_quality_score (0–100), and brand if available.
    Score logic: each section present contributes equal weight.
    """
    if "description" not in df.columns or df.empty:
        return pd.DataFrame()

    text = df["description"].fillna("").astype(str)
    result = pd.DataFrame(index=df.index)

    for section, patterns in DESC_SECTION_PATTERNS.items():
        combined = "|".join(f"(?:{p})" for p in patterns)
        result[section] = text.str.contains(
            combined, flags=re.IGNORECASE, regex=True
        ).astype(int)

    n_sections = len(DESC_SECTION_PATTERNS)
    result["description_quality_score"] = (
        result[list(DESC_SECTION_PATTERNS.keys())].sum(axis=1) / n_sections * 100
    ).round(1)

    if "brand" in df.columns:
        result["brand"] = df["brand"].values

    return result


def compute_feature_benefit_balance(
    df: pd.DataFrame, text_col: str = "description"
) -> pd.DataFrame:
    """Estimate whether each listing leans toward feature or benefit language.

    Uses pattern matching heuristics (not NLP). Returns a DataFrame with:
      - feature_hits: number of feature-language patterns matched
      - benefit_hits: number of benefit-language patterns matched
      - orientation:  'Feature-Led', 'Benefit-Led', 'Balanced', or 'No Signal'

    Threshold: one side must be 1.5× the other to be classified as 'led'.
    """
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


def build_differentiation_insights(
    claims_df: pd.DataFrame, total_listings: int
) -> dict:
    """Categorize claims by market saturation level.

    Returns a dict with three lists:
      'saturated'       — claims in ≥60% of listings (market standard)
      'differentiating' — claims in 20–59% of listings (notable but not universal)
      'underused'       — claims present but in <20% of listings (white space)

    Each entry is {'claim': str, 'pct': float, 'listings': int}.
    """
    if claims_df.empty or total_listings == 0:
        return {"saturated": [], "differentiating": [], "underused": []}

    saturated, differentiating, underused = [], [], []
    for _, row in claims_df.iterrows():
        pct = row["listings"] / total_listings * 100
        entry = {
            "claim":    row["claim"],
            "pct":      round(pct, 1),
            "listings": int(row["listings"]),
        }
        if pct >= 60:
            saturated.append(entry)
        elif pct >= 20:
            differentiating.append(entry)
        elif row["listings"] > 0:
            underused.append(entry)

    return {
        "saturated":       sorted(saturated,       key=lambda x: -x["pct"]),
        "differentiating": sorted(differentiating, key=lambda x: -x["pct"]),
        "underused":       sorted(underused,       key=lambda x: -x["pct"]),
    }


def extract_section_examples(
    df: pd.DataFrame,
    text_col: str = "description",
    n_examples: int = 3,
    context_chars: int = 60,
) -> dict:
    """Extract short real-text examples from the dataset for each DESC_SECTION_PATTERNS section.

    For each section, searches actual descriptions and returns a list of short
    matching snippets so clients see real product language rather than abstract
    pattern descriptions.

    Args:
        df:            DataFrame containing the text column.
        text_col:      Column to search (usually "description").
        n_examples:    Maximum examples to return per section.
        context_chars: Max characters to include after the match start.

    Returns:
        Dict mapping section name → list of example strings (may be empty).
    """
    if text_col not in df.columns or df.empty:
        return {section: [] for section in DESC_SECTION_PATTERNS}

    text_series = df[text_col].fillna("").astype(str)
    results: dict = {}

    for section, patterns in DESC_SECTION_PATTERNS.items():
        combined = "|".join(f"(?:{p})" for p in patterns)
        found: List[str] = []

        for text in text_series:
            if len(found) >= n_examples:
                break
            m = re.search(combined, text, flags=re.IGNORECASE)
            if not m:
                continue
            # Grab the match + a window of surrounding text, then clean up
            start = max(0, m.start() - 3)
            end   = min(len(text), m.start() + context_chars)
            snippet = text[start:end].strip()
            snippet = re.sub(r"\s+", " ", snippet)          # collapse whitespace
            snippet = re.sub(r"^[^a-zA-Z#]+", "", snippet)  # strip leading junk
            if len(snippet) > 80:
                snippet = snippet[:77] + "…"
            if snippet and snippet not in found:
                found.append(snippet)

        results[section] = found

    return results


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    """Load the cleaned CSV, normalise column types, and apply data-quality filters.

    Decorated with @st.cache_data so it runs once per session, not on every
    widget interaction.

    Data-quality steps applied here (before any analysis):
      1. Coerce numeric columns.
      2. Rename thumbnailimage → thumbnailImage for readability.
      3. Filter to baby products using breadcrumbs (keeps the dataset relevant).
      4. Remove price outliers — only realistic baby-product prices are kept.
    """
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    # ── 1. Coerce numeric columns up front — avoids repeated per-tab coercions
    for col in ["stars", "reviewsCount", "price_value", "title_word_count", "description_length"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 2. Rename thumbnail column produced by clean_data.py (all lowercase)
    if "thumbnailimage" in df.columns:
        df.rename(columns={"thumbnailimage": "thumbnailImage"}, inplace=True)

    # ── 2b. Normalize brand names — consolidates variants like "Dr. Brown's" /
    # "Dr Browns" / "dr brown" into a single canonical name so brand charts
    # and filters are clean. Unknown/empty brands become "Unknown".
    if "brand" in df.columns:
        df["brand"] = df["brand"].apply(normalize_brand)

    # ── 3. Restrict to baby products only.
    # The breadcrumbs column (may be stored as "breadcrumbs" after clean_data.py
    # lowercases headers) contains the full category path, e.g.
    # "Baby Products > Feeding > ...".  Keep any row whose breadcrumbs contain
    # "Baby" (case-insensitive).  Rows with no breadcrumbs are kept to avoid
    # silently dropping products that simply have missing metadata.
    bc_col = None
    for candidate in ["breadcrumbs", "breadCrumbs", "Breadcrumbs"]:
        if candidate in df.columns:
            bc_col = candidate
            break
    if bc_col is not None:
        is_baby = df[bc_col].astype(str).str.contains("baby", case=False, na=True)
        df = df[is_baby].copy()

    # ── 4. Remove price outliers.
    # Prices outside $3–$100 are almost certainly data errors (e.g. 8086.46)
    # and would skew every price-related chart.  Rows with a missing price are
    # kept so as not to exclude valid products that simply lack price data.
    if "price_value" in df.columns:
        df = df[df["price_value"].isna() | df["price_value"].between(3, 100)].copy()

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR FILTERS
# Returns a filtered slice of the full DataFrame used by all tabs.
# ══════════════════════════════════════════════════════════════════════════════

def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filter widgets and return the filtered DataFrame.

    Filters: brand (multiselect), category_root (multiselect),
             price range (slider), rating range (slider).
    """
    st.sidebar.header("Filters")
    filtered = df.copy()

    # Brand multiselect
    if "brand" in df.columns:
        brand_options = sorted(df["brand"].dropna().astype(str).unique())
        selected_brands = st.sidebar.multiselect("Brand", brand_options, default=brand_options)
        filtered = filtered[filtered["brand"].astype(str).isin(selected_brands)]

    # Category multiselect
    if "category_root" in df.columns:
        cat_options = sorted(df["category_root"].dropna().astype(str).unique())
        selected_cats = st.sidebar.multiselect("Category", cat_options, default=cat_options)
        filtered = filtered[filtered["category_root"].astype(str).isin(selected_cats)]

    # Price range slider
    if "price_value" in df.columns:
        price_clean = df["price_value"].dropna()
        if not price_clean.empty:
            min_p, max_p = float(price_clean.min()), float(price_clean.max())
            if min_p < max_p:
                price_range = st.sidebar.slider(
                    "Price Range ($)",
                    min_value=min_p, max_value=max_p,
                    value=(min_p, max_p), step=0.5,
                )
                # Preserve rows with a missing price (don't exclude them silently)
                filtered = filtered[
                    filtered["price_value"].isna()
                    | filtered["price_value"].between(price_range[0], price_range[1])
                ]

    # Rating range slider
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
# TAB: OVERVIEW
# Lightweight snapshot — KPIs + brand distribution only.
# ══════════════════════════════════════════════════════════════════════════════

def tab_overview(df: pd.DataFrame) -> None:
    """Render the Overview tab: KPI metrics, summary callout, and brand distribution charts."""
    st.header("Market Overview")
    st.caption("A high-level snapshot of the competitive landscape across all filtered products.")

    # ── KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Products", f"{len(df):,}")

    avg_rating = df["stars"].mean() if "stars" in df.columns else np.nan
    c2.metric("Avg Rating", f"{avg_rating:.2f} \u2605" if pd.notna(avg_rating) else "N/A")

    total_reviews = df["reviewsCount"].sum() if "reviewsCount" in df.columns else np.nan
    c3.metric("Total Reviews", f"{int(total_reviews):,}" if pd.notna(total_reviews) else "N/A")

    avg_price = df["price_value"].mean() if "price_value" in df.columns else np.nan
    c4.metric("Avg Price", f"${avg_price:.2f}" if pd.notna(avg_price) else "N/A")

    # ── Market snapshot callout ─────────────────────────────────────────────
    if "brand" in df.columns:
        n_brands = df["brand"].nunique()
        top_brand = df["brand"].value_counts().idxmax() if len(df) > 0 else "N/A"
        top_brand_pct = round(df["brand"].value_counts().max() / max(len(df), 1) * 100)

        callout_parts = [f"**{n_brands} brands** compete in this dataset"]
        callout_parts.append(
            f"**{top_brand}** leads on product count ({top_brand_pct}% of listings)"
        )
        if pd.notna(avg_rating):
            callout_parts.append(f"average category rating is **{avg_rating:.1f} \u2605**")

        st.info("  ·  ".join(callout_parts))

    st.divider()

    # ── Brand distribution
    st.subheader("Brand Landscape")
    st.caption(
        "Which brands have the most products listed, and which have accumulated the most "
        "customer reviews — a proxy for market presence and validated demand."
    )
    if "brand" not in df.columns:
        st.info("No 'brand' column found.")
        return

    left, right = st.columns(2)

    with left:
        brand_count = (
            df["brand"].value_counts().head(15)
            .rename_axis("brand").reset_index(name="products")
        )
        fig = px.bar(
            brand_count, x="brand", y="products",
            title="Top 15 Brands by Number of Products",
            color="products", color_continuous_scale="Blues",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Products Listed", xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if "reviewsCount" in df.columns:
            brand_rev = (
                df.groupby("brand", dropna=False)["reviewsCount"]
                .sum().sort_values(ascending=False).head(15).reset_index()
            )
            fig2 = px.bar(
                brand_rev, x="brand", y="reviewsCount",
                title="Top 15 Brands by Total Review Volume",
                color="reviewsCount", color_continuous_scale="Oranges",
            )
            fig2.update_layout(showlegend=False, xaxis_title="", yaxis_title="Total Reviews", xaxis_tickangle=-30)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("reviewsCount column not found.")



# ══════════════════════════════════════════════════════════════════════════════
# TAB: TITLE INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

def tab_title_intelligence(df: pd.DataFrame) -> None:
    """Render the Title Intelligence tab.

    Sections:
        1. Top keywords / phrases — phrase-aware bar chart + frequency table
        2. Title length distribution — histogram with median marker
        3. Top listing title examples — top-15 sorted by reviewsCount
        4. Title Structure & Quality
    """
    st.header("Title Intelligence")
    st.caption(
        "Uncover patterns in how competitors write product titles — "
        "keyword choices, phrase usage, length strategy, and quality scoring. "
        "Analysis is scoped to **baby bottles and bottle nipples** only."
    )

    # Apply category filter — analysis scoped to bottles and nipples only
    df = filter_bottles_and_nipples(df)
    if df.empty:
        st.warning("No baby bottle or nipple listings found with the current filters.")
        return

    n_filtered = len(df)
    st.caption(f"Analysing **{n_filtered:,} baby bottle / nipple listings**.")

    # Prefer the lowercased clean_title for cleaner tokenization
    title_col = "clean_title" if "clean_title" in df.columns else "title"

    # ── Section 1: Top title keywords & phrases ───────────────────────────────
    st.subheader("Top Title Keywords & Phrases")
    st.caption(
        "The most common keywords and phrases used across product titles. "
        "Multi-word phrases like 'Anti Colic' or 'BPA Free' are grouped as single concepts "
        "so they appear as meaningful signals rather than split individual words. "
        "Brand names are excluded — this focuses on product features, benefits, and claims."
    )

    if title_col in df.columns:
        kw_df = extract_keyword_frequencies(
            df[title_col],
            phrase_map=TITLE_PHRASE_MAP,
            stopwords=TITLE_STOPWORDS,
            top_n=20,
        )

        if not kw_df.empty:
            fig_kw = px.bar(
                kw_df,
                x="keyword",
                y="frequency",
                title="Top Title Keywords & Phrases (Baby Bottles & Nipples)",
                color="frequency",
                color_continuous_scale="Teal",
            )
            fig_kw.update_layout(
                showlegend=False,
                xaxis_title="Keyword / Phrase",
                yaxis_title="Number of Titles",
                xaxis_tickangle=-35,
            )
            st.plotly_chart(fig_kw, use_container_width=True)

            st.dataframe(
                kw_df.rename(columns={"keyword": "Keyword / Phrase", "frequency": "Titles Containing It"}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No title keywords found after filtering stopwords.")
    else:
        st.info("Title column not found in the dataset.")

    st.divider()

    # ── Section 2: Title length distribution ──────────────────────────────────
    st.subheader("Title Length Distribution")
    if "title_word_count" in df.columns:
        median_wc = df["title_word_count"].median()
        fig_len = px.histogram(
            df, x="title_word_count", nbins=30,
            title="Distribution of Title Word Count",
            color_discrete_sequence=["#4A90D9"],
        )
        fig_len.update_layout(xaxis_title="Word Count", yaxis_title="Number of Products")
        # Median reference line helps clients quickly benchmark their own titles
        fig_len.add_vline(
            x=median_wc, line_dash="dash", line_color="red",
            annotation_text=f"Median: {median_wc:.0f} words",
        )
        st.plotly_chart(fig_len, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Median Title Length", f"{int(median_wc)} words")
        c2.metric("Avg Title Length", f"{df['title_word_count'].mean():.1f} words")
        c3.metric("Max Title Length", f"{int(df['title_word_count'].max())} words")
    else:
        st.info("title_word_count column not found.")

    st.divider()

    # ── Section 3: Top listing title examples ─────────────────────────────────
    st.subheader("Top-Performing Title Examples")
    st.caption(
        "The 15 highest-reviewed listings in the dataset — "
        "titles with the most customer validation."
    )

    if "title" in df.columns:
        example_cols = [
            c for c in ["title", "brand", "reviewsCount", "stars", "price_value"]
            if c in df.columns
        ]
        examples = (
            df[example_cols]
            .dropna(subset=["title"])
            .sort_values("reviewsCount", ascending=False)
            .head(15)
            .reset_index(drop=True)
        )
        col_cfg: dict = {}
        if "stars" in examples.columns:
            col_cfg["stars"] = st.column_config.NumberColumn("Rating", format="%.1f \u2605")
        if "reviewsCount" in examples.columns:
            col_cfg["reviewsCount"] = st.column_config.NumberColumn("Reviews", format="%d")
        if "price_value" in examples.columns:
            col_cfg["price_value"] = st.column_config.NumberColumn("Price", format="$%.2f")

        st.dataframe(examples, column_config=col_cfg, use_container_width=True, hide_index=True)
    else:
        st.info("Title column not found.")

    st.divider()

    # ── Section 4: Common Title Attributes ────────────────────────────────────
    st.subheader("Common Title Attributes")
    st.caption(
        "Which product attributes appear most frequently in titles across the market? "
        "This shows what sellers prioritise — and what is often left out — "
        "across flow rate, size, material, quantity, anti-colic feature, and more."
    )

    title_scores_df = compute_title_quality_scores(df)

    if not title_scores_df.empty:
        total = max(len(df), 1)
        comp_coverage_rows = []
        for component in TITLE_COMPONENTS:
            if component in title_scores_df.columns:
                count = int(title_scores_df[component].sum())
                comp_coverage_rows.append({
                    "Title Attribute":     component,
                    "Titles Including It": count,
                    "% of Titles":         round(count / total * 100, 1),
                })
        comp_cov_df = (
            pd.DataFrame(comp_coverage_rows)
            .sort_values("% of Titles", ascending=False)
            .reset_index(drop=True)
        )

        fig_comp_cov = px.bar(
            comp_cov_df, x="Title Attribute", y="% of Titles",
            title="How Often Each Attribute Appears in Product Titles",
            color="% of Titles", color_continuous_scale="Teal",
        )
        fig_comp_cov.update_layout(
            showlegend=False, xaxis_title="", yaxis_title="% of Titles", xaxis_tickangle=-20,
        )
        st.plotly_chart(fig_comp_cov, use_container_width=True)
        st.dataframe(comp_cov_df, use_container_width=True, hide_index=True)
    else:
        st.info("Unable to analyse title attributes — title column not found.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB: DESCRIPTION INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

def tab_description_intelligence(df: pd.DataFrame) -> None:
    """Render the Description Intelligence tab.

    Sections:
        1. Top messaging themes
        2. Common product claims
        3. Description length distribution
        4. Messaging category coverage
        5. Description quality score
        6. Feature vs benefit balance
        7. Competitive differentiation opportunities
    """
    st.header("Description Intelligence")
    st.caption(
        "What language and persuasion tactics are competitors using? "
        "Find the themes, claims, and quality patterns that define the category. "
        "Analysis is scoped to **baby bottles and bottle nipples** only."
    )

    # Apply category filter — analysis scoped to bottles and nipples only
    df = filter_bottles_and_nipples(df)
    if df.empty:
        st.warning("No baby bottle or nipple listings found with the current filters.")
        return

    st.caption(f"Analysing **{len(df):,} baby bottle / nipple listings**.")

    if "description" not in df.columns:
        st.error("No 'description' column found. Re-run scripts/clean_data.py.")
        return

    # ── Section 1: Top messaging themes ───────────────────────────────────────
    st.subheader("Top Messaging Themes")
    st.caption(
        "The most common language used across competitor descriptions. "
        "Multi-word phrases are treated as single concepts — so 'Anti Colic' and "
        "'BPA Free' count as whole ideas rather than scattered individual words."
    )

    desc_kw_df = extract_keyword_frequencies(
        df["description"],
        phrase_map=DESC_PHRASE_MAP,
        stopwords=DESC_STOPWORDS,
        top_n=25,
    )

    if not desc_kw_df.empty:
        fig_themes = px.bar(
            desc_kw_df, x="keyword", y="frequency",
            title="Most Common Themes in Product Descriptions",
            color="frequency", color_continuous_scale="Greens",
        )
        fig_themes.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Number of Descriptions",
            xaxis_tickangle=-35,
        )
        st.plotly_chart(fig_themes, use_container_width=True)

        st.dataframe(
            desc_kw_df.rename(columns={"keyword": "Messaging Theme", "frequency": "Descriptions Containing It"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No description themes found after filtering stopwords.")

    st.divider()

    # ── Section 2: Common product claims ──────────────────────────────────────
    st.subheader("Common Product Claims")
    st.caption(
        "How many listings explicitly make each claim in their description? "
        "Each listing is counted once per claim, regardless of repetition."
    )

    claims_df = build_claim_presence(df, "description", CLAIM_MAP)
    active_claims = claims_df[claims_df["listings"] > 0].copy()

    if not active_claims.empty:
        fig_claims = px.bar(
            active_claims, x="claim", y="listings",
            title="How Often Each Product Claim Appears Across Listings",
            color="listings", color_continuous_scale="Reds",
        )
        fig_claims.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Number of Listings",
            xaxis_tickangle=-20,
        )
        st.plotly_chart(fig_claims, use_container_width=True)

        st.dataframe(
            active_claims.rename(columns={"claim": "Product Claim", "listings": "Listings", "% of Products": "% of All Products"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No product claims detected across descriptions.")

    st.divider()

    # ── Section 3: Description length distribution ─────────────────────────────
    st.subheader("Description Length Distribution")
    if "description_length" in df.columns:
        median_len = df["description_length"].median()
        fig_len = px.histogram(
            df, x="description_length", nbins=30,
            title="Description Length Distribution",
            color_discrete_sequence=["#6B8E23"],
        )
        fig_len.update_layout(xaxis_title="Character Count", yaxis_title="Number of Products")
        fig_len.add_vline(
            x=median_len, line_dash="dash", line_color="red",
            annotation_text=f"Median: {int(median_len):,} chars",
        )
        st.plotly_chart(fig_len, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Median Length", f"{int(median_len):,} chars")
        c2.metric("Avg Length", f"{int(df['description_length'].mean()):,} chars")
        c3.metric("Max Length", f"{int(df['description_length'].max()):,} chars")
    else:
        st.info("description_length column not found.")

    st.divider()

    # ── Section 4: Messaging Category Coverage ─────────────────────────────────
    st.subheader("Messaging Category Coverage")
    st.caption(
        "Which strategic messaging categories do competitor descriptions cover? "
        "A category is 'present' if at least one relevant phrase appears in the description."
    )

    # ── Examples expander — shown FIRST so clients understand the categories
    # before viewing the analysis charts.
    with st.expander("What does each messaging category mean? (click to expand)", expanded=False):
        st.caption(
            "Each category is detected when any matching phrase appears in the description. "
            "Here are typical examples from each category:"
        )
        for cat, examples in MESSAGING_CATEGORY_EXAMPLES.items():
            st.markdown(f"**{cat}**")
            for ex in examples:
                st.markdown(f"  - *{ex}*")
            st.markdown("")

    # ── 4a: Social proof signals ───────────────────────────────────────────────
    st.markdown("**Trust & Social Proof Signals**")
    st.caption(
        "How often do listings use specific trust phrases — "
        "such as '#1 Brand', 'Recommended by Moms', or 'Clinically Tested'?"
    )
    sp_rows = []
    for phrase, pattern in SOCIAL_PROOF_PATTERNS.items():
        count = int(
            df["description"].fillna("").astype(str)
            .str.contains(pattern, flags=re.IGNORECASE, regex=True).sum()
        )
        sp_rows.append({
            "Social Proof Phrase": phrase,
            "Listings": count,
            "% of Products": round(count / max(len(df), 1) * 100, 1),
        })
    sp_df = pd.DataFrame(sp_rows).sort_values("Listings", ascending=False).reset_index(drop=True)

    sp_active = sp_df[sp_df["Listings"] > 0]
    if not sp_active.empty:
        fig_sp = px.bar(
            sp_active, x="Social Proof Phrase", y="% of Products",
            title="Social Proof Signal Frequency (% of Listings)",
            color="% of Products", color_continuous_scale="Purples",
        )
        fig_sp.update_layout(
            showlegend=False, xaxis_tickangle=-25,
            xaxis_title="", yaxis_title="% of Listings",
        )
        st.plotly_chart(fig_sp, use_container_width=True)
        st.dataframe(sp_df, use_container_width=True, hide_index=True)
    else:
        st.info("No social proof signals detected in descriptions with current filters.")

    # ── 4b: Messaging category coverage across all listings ────────────────────
    st.markdown("**Category Coverage Across All Listings**")
    st.caption(
        "What share of listings use language from each messaging category? "
        "Categories with lower coverage may represent opportunities for stronger positioning."
    )
    cat_rows = []
    for category, patterns in MESSAGING_CATEGORIES.items():
        combined = "|".join(f"(?:{p})" for p in patterns)
        count = int(
            df["description"].fillna("").astype(str)
            .str.contains(combined, flags=re.IGNORECASE, regex=True).sum()
        )
        cat_rows.append({
            "Category": category,
            "Listings": count,
            "% of Products": round(count / max(len(df), 1) * 100, 1),
        })
    cat_df = (
        pd.DataFrame(cat_rows)
        .sort_values("% of Products", ascending=False)
        .reset_index(drop=True)
    )
    fig_cat = px.bar(
        cat_df, x="Category", y="% of Products",
        title="Messaging Category Coverage Across All Listings",
        color="% of Products", color_continuous_scale="Teal",
    )
    fig_cat.update_layout(
        showlegend=False, xaxis_title="", yaxis_title="% of Listings", xaxis_tickangle=-20,
    )
    st.plotly_chart(fig_cat, use_container_width=True)
    st.dataframe(cat_df, use_container_width=True, hide_index=True)

    # ── 4c: Messaging breadth score distribution ───────────────────────────────
    st.markdown("**Messaging Breadth Score Distribution**")
    st.caption(
        "Each listing is scored 0–100 based on how many of the six messaging categories "
        "its description covers. A higher score means broader, more complete messaging."
    )
    scores_df = build_messaging_category_scores(df, "description")
    avg_score = scores_df["claim_coverage_score"].mean()

    fig_score = px.histogram(
        scores_df, x="claim_coverage_score", nbins=20,
        title="Messaging Breadth Score Distribution",
        color_discrete_sequence=["#7B68EE"],
    )
    fig_score.update_layout(
        xaxis_title="Messaging Breadth Score (0–100)", yaxis_title="Number of Listings",
    )
    fig_score.add_vline(
        x=avg_score, line_dash="dash", line_color="red",
        annotation_text=f"Avg: {avg_score:.1f}",
    )
    st.plotly_chart(fig_score, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Avg Messaging Breadth Score", f"{avg_score:.1f} / 100")
    fully_covered = int((scores_df["claim_coverage_score"] == 100).sum())
    c2.metric("Listings Covering All 6 Categories", f"{fully_covered:,}")

    st.divider()

    # ── Section 5: Tommee Tippee vs Competitor Messaging ──────────────────────
    st.subheader("Tommee Tippee vs Competitor Messaging")
    st.caption(
        "Tommee Tippee's description messaging compared against all other competitors. "
        "Where competitor bars are taller, Tommee Tippee's content lags — "
        "these are the clearest opportunities to strengthen copy."
    )

    if "brand" not in df.columns:
        st.info("Brand column required for Tommee Tippee comparison.")
    else:
        # Identify TT vs competitors within the already-filtered df
        is_tt   = df["brand"].astype(str).str.contains("tommee", case=False, na=False)
        tt_sub  = df[is_tt]
        comp_sub = df[~is_tt]

        if tt_sub.empty:
            st.info(
                "No Tommee Tippee products found in the current filters. "
                "Re-enable Tommee Tippee in the sidebar Brand filter to see this section."
            )
        else:
            # ── 5a: Grouped messaging category comparison chart ────────────────
            comp_data = build_messaging_comparison(
                tt_sub, comp_sub, "description",
                label_a="Tommee Tippee", label_b="Competitors",
            )
            fig_comp = px.bar(
                comp_data, x="Category", y="% of Listings", color="Group",
                barmode="group",
                title="Messaging Category Coverage: Tommee Tippee vs Competitors",
                color_discrete_map={"Tommee Tippee": "#E8552A", "Competitors": "#4A90D9"},
            )
            fig_comp.update_layout(
                xaxis_title="", yaxis_title="% of Listings",
                xaxis_tickangle=-20, legend_title="",
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            # ── 5b: Claim coverage score comparison ───────────────────────────
            tt_scores   = build_messaging_category_scores(tt_sub,   "description")
            comp_scores = build_messaging_category_scores(comp_sub, "description")

            tt_avg   = tt_scores["claim_coverage_score"].mean()
            comp_avg = comp_scores["claim_coverage_score"].mean() if len(comp_sub) > 0 else 0.0
            delta    = tt_avg - comp_avg

            c1, c2, c3 = st.columns(3)
            c1.metric("TT Avg Coverage Score",         f"{tt_avg:.1f} / 100")
            c2.metric("Competitor Avg Coverage Score", f"{comp_avg:.1f} / 100")
            c3.metric(
                "TT vs Competitor Gap",
                f"{delta:+.1f} pts",
                help="Positive = Tommee Tippee covers more categories on average. "
                     "Negative = competitors are more comprehensive.",
            )

            # ── 5c: Per-category gap table ────────────────────────────────────
            st.markdown("**Category-Level Opportunity Table**")
            st.caption(
                "Positive gap = competitors use this category more than Tommee Tippee "
                "→ a clear opportunity to strengthen messaging."
            )
            gap_rows = []
            tt_n   = max(len(tt_sub), 1)
            comp_n = max(len(comp_sub), 1)
            for category, patterns in MESSAGING_CATEGORIES.items():
                combined = "|".join(f"(?:{p})" for p in patterns)
                tt_pct = round(
                    tt_sub["description"].fillna("").astype(str)
                    .str.contains(combined, flags=re.IGNORECASE, regex=True).sum()
                    / tt_n * 100, 1,
                )
                comp_pct = round(
                    comp_sub["description"].fillna("").astype(str)
                    .str.contains(combined, flags=re.IGNORECASE, regex=True).sum()
                    / comp_n * 100, 1,
                )
                gap_rows.append({
                    "Category":        category,
                    "TT %":            tt_pct,
                    "Competitor %":    comp_pct,
                    "Gap (Comp \u2212 TT)": round(comp_pct - tt_pct, 1),
                })
            gap_tbl = (
                pd.DataFrame(gap_rows)
                .sort_values("Gap (Comp \u2212 TT)", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(gap_tbl, use_container_width=True, hide_index=True)

    st.divider()

    # ── Section 6: Description Quality Score ──────────────────────────────────
    st.subheader("Description Quality Score")
    st.caption(
        "How well-structured are competitor descriptions? "
        "Each listing is evaluated across seven content sections. "
        "A higher score means more complete, persuasive copy."
    )

    # Pull real examples from the dataset before scoring — shown in expander
    section_examples = extract_section_examples(df, text_col="description", n_examples=3)
    with st.expander("What does each content section mean? (real examples from the dataset)", expanded=False):
        st.caption("Each section is detected when any matching phrase appears in the description.")
        for section, ex_list in section_examples.items():
            st.markdown(f"**{section}**")
            if ex_list:
                for ex in ex_list:
                    st.markdown(f'  - *"{ex}"*')
            else:
                st.markdown("  *No examples found in the current dataset.*")
            st.markdown("")

    desc_quality_df = compute_description_quality_scores(df)
    if not desc_quality_df.empty:
        avg_dq = desc_quality_df["description_quality_score"].mean()

        # Section coverage bar chart
        total = max(len(df), 1)
        section_coverage_rows = []
        for section in DESC_SECTION_PATTERNS:
            if section in desc_quality_df.columns:
                count = int(desc_quality_df[section].sum())
                section_coverage_rows.append({
                    "Description Section":     section,
                    "Listings Including It":   count,
                    "% of Listings":           round(count / total * 100, 1),
                })
        sec_cov_df = (
            pd.DataFrame(section_coverage_rows)
            .sort_values("% of Listings", ascending=False)
            .reset_index(drop=True)
        )

        fig_dq_cov = px.bar(
            sec_cov_df, x="Description Section", y="% of Listings",
            title="Description Section Coverage — Which Elements Are Most Common?",
            color="% of Listings", color_continuous_scale="Greens",
        )
        fig_dq_cov.update_layout(
            showlegend=False, xaxis_title="", yaxis_title="% of Listings", xaxis_tickangle=-20,
        )
        st.plotly_chart(fig_dq_cov, use_container_width=True)

        # KPIs
        least_covered_sec = sec_cov_df.iloc[-1] if not sec_cov_df.empty else None
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Description Quality Score", f"{avg_dq:.0f} / 100")
        if least_covered_sec is not None:
            c2.metric(
                "Most Missing Section",
                least_covered_sec["Description Section"],
                help=f"Only {least_covered_sec['% of Listings']:.0f}% of listings include this section.",
            )
        strong_descs = int((desc_quality_df["description_quality_score"] >= 57).sum())
        c3.metric(
            "Listings Covering 4+ Sections",
            f"{strong_descs:,}",
            help="Listings that include at least 4 of the 7 tracked content sections.",
        )

        # Quality score distribution
        fig_dqs_hist = px.histogram(
            desc_quality_df, x="description_quality_score", nbins=15,
            title="Description Quality Score Distribution",
            color_discrete_sequence=["#6B8E23"],
        )
        fig_dqs_hist.update_layout(
            xaxis_title="Description Quality Score (0–100)", yaxis_title="Number of Listings",
        )
        fig_dqs_hist.add_vline(
            x=avg_dq, line_dash="dash", line_color="red",
            annotation_text=f"Avg: {avg_dq:.0f}",
        )
        st.plotly_chart(fig_dqs_hist, use_container_width=True)

        # Brand-level description quality
        if "brand" in desc_quality_df.columns:
            brand_dq = (
                desc_quality_df.groupby("brand")["description_quality_score"]
                .mean().round(1).sort_values(ascending=False)
                .reset_index().head(15)
                .rename(columns={"brand": "Brand", "description_quality_score": "Avg Description Quality Score"})
            )
            fig_bdq = px.bar(
                brand_dq, x="Brand", y="Avg Description Quality Score",
                title="Average Description Quality Score by Brand",
                color="Avg Description Quality Score", color_continuous_scale="Greens",
            )
            fig_bdq.update_layout(
                showlegend=False, xaxis_title="", yaxis_title="Score (0–100)", xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_bdq, use_container_width=True)

        st.dataframe(sec_cov_df, use_container_width=True, hide_index=True)
    else:
        st.info("Unable to compute description quality scores — description column not found.")

    st.divider()

    # ── Section 7: Feature vs Benefit Balance ─────────────────────────────────
    st.subheader("Feature vs Benefit Balance")
    st.caption(
        "Are competitor listings primarily telling customers **what** a product has (features), "
        "or **what** it does for them (benefits)? "
        "Benefit-led copy tends to be more persuasive and emotionally resonant. "
        "This is a practical estimate based on language patterns in the description."
    )

    fb_df = compute_feature_benefit_balance(df, "description")
    if not fb_df.empty:
        orient_counts = fb_df["orientation"].value_counts().reset_index()
        orient_counts.columns = ["Orientation", "Listings"]

        col_left, col_right = st.columns(2)
        with col_left:
            fig_fb = px.bar(
                orient_counts, x="Orientation", y="Listings",
                title="Listing Messaging Orientation",
                color="Orientation",
                color_discrete_map={
                    "Feature-Led":  "#4A90D9",
                    "Benefit-Led":  "#2ECC71",
                    "Balanced":     "#F39C12",
                    "No Signal":    "#BDC3C7",
                },
            )
            fig_fb.update_layout(showlegend=False, xaxis_title="", yaxis_title="Number of Listings")
            st.plotly_chart(fig_fb, use_container_width=True)

        with col_right:
            total_signal = fb_df[fb_df["orientation"] != "No Signal"].shape[0]
            benefit_led  = int((fb_df["orientation"] == "Benefit-Led").sum())
            feature_led  = int((fb_df["orientation"] == "Feature-Led").sum())
            balanced     = int((fb_df["orientation"] == "Balanced").sum())
            st.markdown("**Breakdown**")
            st.metric("Benefit-Led Listings", f"{benefit_led:,}")
            st.metric("Feature-Led Listings", f"{feature_led:,}")
            st.metric("Balanced Listings",    f"{balanced:,}")
            if total_signal > 0:
                benefit_pct = round(benefit_led / total_signal * 100)
                st.caption(
                    f"**{benefit_pct}%** of listings with detectable signal "
                    f"lean toward benefit language."
                )

        # Brand-level breakdown
        if "brand" in fb_df.columns:
            brand_orient = (
                fb_df[fb_df["orientation"] != "No Signal"]
                .groupby(["brand", "orientation"])
                .size().reset_index(name="count")
            )
            if not brand_orient.empty:
                fig_fb_brand = px.bar(
                    brand_orient, x="brand", y="count", color="orientation",
                    barmode="stack",
                    title="Feature vs Benefit Orientation by Brand",
                    color_discrete_map={
                        "Feature-Led": "#4A90D9",
                        "Benefit-Led": "#2ECC71",
                        "Balanced":    "#F39C12",
                    },
                )
                fig_fb_brand.update_layout(
                    xaxis_title="", yaxis_title="Listings",
                    xaxis_tickangle=-30, legend_title="Orientation",
                )
                st.plotly_chart(fig_fb_brand, use_container_width=True)
    else:
        st.info("Unable to compute feature/benefit balance — description column not found.")

    st.divider()

    # ── Section 8: Competitive Differentiation Opportunities ──────────────────
    st.subheader("Competitive Differentiation Opportunities")
    st.caption(
        "Not all claims are equal. Claims used by nearly every competitor are the market baseline — "
        "expected but not differentiating. Claims used by fewer listings may represent a real "
        "opportunity to stand out."
    )

    diff_claims_df = build_claim_presence(df, "description", CLAIM_MAP)
    diff_insights  = build_differentiation_insights(diff_claims_df, total_listings=len(df))

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Market Standard (≥60% of listings)**")
        st.caption("Every serious competitor uses these. Table stakes — must have, but not differentiating.")
        if diff_insights["saturated"]:
            for item in diff_insights["saturated"]:
                st.markdown(f"- **{item['claim']}** — {item['pct']}% of listings")
        else:
            st.info("No claims are this widespread in the current dataset.")

    with col_b:
        st.markdown("**Notable Claims (20–59% of listings)**")
        st.caption("Common enough to be meaningful but not universal — a solid differentiator if used well.")
        if diff_insights["differentiating"]:
            for item in diff_insights["differentiating"]:
                st.markdown(f"- **{item['claim']}** — {item['pct']}% of listings")
        else:
            st.info("No claims in this range with current data.")

    with col_c:
        st.markdown("**White Space Claims (<20% of listings)**")
        st.caption("Rarely used but present in the market — potential to stand out if the claim is authentic.")
        if diff_insights["underused"]:
            for item in diff_insights["underused"]:
                st.markdown(f"- **{item['claim']}** — {item['pct']}% of listings")
        else:
            st.info("No underused claims detected — all active claims are broadly used.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB: IMAGE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def tab_image_analysis(df: pd.DataFrame) -> None:
    """Render the Image Analysis tab.

    Sections:
        1. Thumbnail gallery — 4 per row, sorted by reviewsCount, max 60,
           brand filter, brand + truncated title caption below each image
        2. Brand image comparison — one hero thumbnail per brand
    """
    st.header("Image Analysis")
    st.caption(
        "Browse competitor product thumbnails to study visual presentation strategies "
        "— colors, angles, lifestyle vs. white-background, text overlays, and more."
    )

    img_col = "thumbnailImage" if "thumbnailImage" in df.columns else None
    if img_col is None:
        st.error("No thumbnail image column found. Re-run scripts/clean_data.py.")
        return

    # Only keep rows with valid HTTP image URLs
    img_df = df[df[img_col].notna() & df[img_col].astype(str).str.startswith("http")].copy()

    # Sort by review count descending — most validated products shown first
    if "reviewsCount" in img_df.columns:
        img_df = img_df.sort_values("reviewsCount", ascending=False)

    if img_df.empty:
        st.warning("No valid image URLs found in the dataset.")
        return

    # ── Section 1: Thumbnail gallery ──────────────────────────────────────────
    st.subheader("Thumbnail Gallery")

    # Brand filter scoped to this tab (independent of sidebar)
    if "brand" in img_df.columns:
        brand_list = ["All"] + sorted(img_df["brand"].dropna().unique().tolist())
        selected_brand = st.selectbox("Filter by Brand", brand_list, key="gallery_brand")
        if selected_brand != "All":
            img_df = img_df[img_df["brand"] == selected_brand]

    # Cap displayed images to avoid a slow page load
    max_images = st.slider("Max images to display", min_value=12, max_value=60, value=40, step=4)
    display_df = img_df.head(max_images).reset_index(drop=True)

    images_per_row = 4  # fixed at 4 for consistent, readable grid

    for row_start in range(0, len(display_df), images_per_row):
        row_slice = display_df.iloc[row_start: row_start + images_per_row]
        cols = st.columns(images_per_row)
        for col, (_, product) in zip(cols, row_slice.iterrows()):
            with col:
                try:
                    st.image(str(product[img_col]), use_container_width=True)
                except Exception:
                    st.write("Image unavailable")

                brand_label = str(product.get("brand", "")) if pd.notna(product.get("brand")) else ""
                raw_title = str(product.get("title", ""))
                # Truncate long titles to keep captions compact
                title_label = raw_title[:55] + "\u2026" if len(raw_title) > 55 else raw_title
                st.caption(f"**{brand_label}**  \n{title_label}")

    st.divider()

    # ── Section 2: Brand image comparison ─────────────────────────────────────
    st.subheader("Brand Image Comparison")
    st.caption(
        "One representative thumbnail per brand (highest-reviewed product). "
        "Quickly audit how each brand styles its primary image."
    )

    if "brand" in img_df.columns:
        # Pick the first row per brand after sorting by reviewsCount (already done above)
        brand_samples = (
            img_df.dropna(subset=[img_col])
            .groupby("brand", as_index=False)
            .first()
        )
        # Ensure URLs are still valid after groupby
        brand_samples = brand_samples[
            brand_samples[img_col].astype(str).str.startswith("http")
        ].head(24)

        cols_per_row = 4
        for row_start in range(0, len(brand_samples), cols_per_row):
            row_slice = brand_samples.iloc[row_start: row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, (_, product) in zip(cols, row_slice.iterrows()):
                with col:
                    try:
                        st.image(str(product[img_col]), use_container_width=True)
                    except Exception:
                        st.write("Image unavailable")
                    st.caption(f"**{product.get('brand', '')}**")
    else:
        st.info("Brand column not available for grouping.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB: PRODUCT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

def tab_product_explorer(df: pd.DataFrame) -> None:
    """Render the Product Explorer tab.

    Layout:
        - Collapsible thumbnail gallery (top 16 by reviewsCount, 4 per row)
        - Clean interactive table: title, brand, stars, reviewsCount, price_value, url
          sorted by reviewsCount descending, with free-text search
    """
    st.header("Product Explorer")
    st.caption(
        "Search and compare individual competitor listings. "
        "Click any URL to open the live Amazon product page."
    )

    # ── Free-text search ──────────────────────────────────────────────────────
    search_term = st.text_input(
        "Search by title or brand",
        placeholder="e.g. anti-colic, Philips Avent, glass bottle",
    )

    # Build the table data (no thumbnail column — shown separately below)
    table_cols = [
        c for c in ["title", "brand", "stars", "reviewsCount", "price_value", "url"]
        if c in df.columns
    ]
    explorer_df = df[table_cols].copy()

    if search_term:
        mask = pd.Series(False, index=explorer_df.index)
        if "title" in explorer_df.columns:
            mask |= explorer_df["title"].astype(str).str.contains(search_term, case=False, na=False)
        if "brand" in explorer_df.columns:
            mask |= explorer_df["brand"].astype(str).str.contains(search_term, case=False, na=False)
        explorer_df = explorer_df[mask]

    # Sort by review count descending — most validated products at the top
    if "reviewsCount" in explorer_df.columns:
        explorer_df = explorer_df.sort_values("reviewsCount", ascending=False)

    st.caption(f"Showing {len(explorer_df):,} products")

    # ── Thumbnail gallery (visual companion, collapsible) ─────────────────────
    img_col = "thumbnailImage" if "thumbnailImage" in df.columns else None
    if img_col:
        # Merge thumbnails back using the filtered explorer index
        thumb_df = df.loc[explorer_df.index, [img_col] + [c for c in ["brand", "title"] if c in df.columns]].copy()
        thumb_df = thumb_df[
            thumb_df[img_col].notna()
            & thumb_df[img_col].astype(str).str.startswith("http")
        ].head(16)  # cap at 16 for quick rendering

        if not thumb_df.empty:
            with st.expander("Thumbnail Preview (top 16 results)", expanded=True):
                cols_per_row = 4
                for row_start in range(0, len(thumb_df), cols_per_row):
                    row_slice = thumb_df.iloc[row_start: row_start + cols_per_row]
                    cols = st.columns(cols_per_row)
                    for col, (_, product) in zip(cols, row_slice.iterrows()):
                        with col:
                            try:
                                st.image(str(product[img_col]), use_container_width=True)
                            except Exception:
                                st.write("N/A")
                            brand_label = str(product.get("brand", "")) if pd.notna(product.get("brand")) else ""
                            st.caption(f"**{brand_label}**")

    # ── Data table ────────────────────────────────────────────────────────────
    col_cfg: dict = {}
    if "url" in explorer_df.columns:
        col_cfg["url"] = st.column_config.LinkColumn("Link", display_text="View on Amazon")
    if "stars" in explorer_df.columns:
        col_cfg["stars"] = st.column_config.NumberColumn("Rating", format="%.1f \u2605")
    if "reviewsCount" in explorer_df.columns:
        col_cfg["reviewsCount"] = st.column_config.NumberColumn("Reviews", format="%d")
    if "price_value" in explorer_df.columns:
        col_cfg["price_value"] = st.column_config.NumberColumn("Price", format="$%.2f")

    st.dataframe(
        explorer_df.reset_index(drop=True),
        column_config=col_cfg,
        use_container_width=True,
        height=600,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB: TOMMEE TIPPEE DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════

def tab_tommee_tippee(df: pd.DataFrame) -> None:
    """Render the Tommee Tippee Deep Dive tab.

    Splits the sidebar-filtered DataFrame into Tommee Tippee listings and
    competitor listings, then analyses both groups across six sections:

        1. Overview KPIs
        2. Title Analysis
        3. Description Analysis
        4. Competitor Gap Analysis (title keywords, description themes, claims)
        5. Image Gallery
        6. Recommendations
    """
    st.header("Tommee Tippee Deep Dive")
    st.caption(
        "A focused analysis of Tommee Tippee's Amazon listing content "
        "compared directly against all other competitors in the filtered dataset. "
        "Analysis is scoped to **baby bottles and bottle nipples** only."
    )

    # ── Scope to bottles and nipples only ─────────────────────────────────────
    df = filter_bottles_and_nipples(df)
    if df.empty:
        st.warning("No baby bottle or nipple listings found with the current filters.")
        return

    # ── Identify Tommee Tippee products ───────────────────────────────────────
    if "brand" not in df.columns:
        st.error("No 'brand' column found in the dataset.")
        return

    is_tt   = df["brand"].astype(str).str.contains("tommee", case=False, na=False)
    tt_df   = df[is_tt].copy()
    comp_df = df[~is_tt].copy()

    if tt_df.empty:
        st.warning(
            "No Tommee Tippee products found in the current filtered data. "
            "If you have deselected Tommee Tippee in the sidebar Brand filter, "
            "please re-enable it to use this tab."
        )
        return

    title_col = "clean_title" if "clean_title" in df.columns else "title"

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1: TOMMEE TIPPEE OVERVIEW
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("1. Brand Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TT Products", f"{len(tt_df):,}")

    tt_avg_rating = tt_df["stars"].mean() if "stars" in tt_df.columns else np.nan
    c2.metric("Avg Rating", f"{tt_avg_rating:.2f} \u2605" if pd.notna(tt_avg_rating) else "N/A")

    tt_avg_price = tt_df["price_value"].mean() if "price_value" in tt_df.columns else np.nan
    c3.metric("Avg Price", f"${tt_avg_price:.2f}" if pd.notna(tt_avg_price) else "N/A")

    tt_total_reviews = tt_df["reviewsCount"].sum() if "reviewsCount" in tt_df.columns else np.nan
    c4.metric("Total Reviews", f"{int(tt_total_reviews):,}" if pd.notna(tt_total_reviews) else "N/A")

    # Context row: how many competitors are in the filtered set
    st.caption(
        f"Comparing **{len(tt_df):,} Tommee Tippee listings** "
        f"against **{len(comp_df):,} competitor listings** "
        f"(after applying sidebar filters)."
    )

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2: TITLE INTELLIGENCE
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("2. Title Intelligence")

    # ── 2a: Title keyword frequency (TT only) ─────────────────────────────
    st.markdown("**Top Phrases in Tommee Tippee Titles**")
    st.caption(
        "Multi-word phrases are treated as single concepts — "
        "the same phrase logic used across the rest of the dashboard."
    )

    if title_col in tt_df.columns:
        tt_title_kw = extract_keyword_frequencies(
            tt_df[title_col], TITLE_PHRASE_MAP, TITLE_STOPWORDS, top_n=20
        )
        if not tt_title_kw.empty:
            fig_tt_kw = px.bar(
                tt_title_kw, x="keyword", y="frequency",
                title="Most Frequent Keywords and Phrases in Tommee Tippee Titles",
                color="frequency", color_continuous_scale="Teal",
            )
            fig_tt_kw.update_layout(
                showlegend=False,
                xaxis_title="Keyword / Phrase", yaxis_title="Frequency", xaxis_tickangle=-35,
            )
            st.plotly_chart(fig_tt_kw, use_container_width=True)
            st.dataframe(
                tt_title_kw.rename(columns={"keyword": "Keyword / Phrase", "frequency": "Frequency"}),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No title keywords found for Tommee Tippee after filtering stopwords.")
    else:
        st.info("Title column not found.")

    # ── 2b: Title length distribution (TT vs competitor median overlay) ────
    st.markdown("**Title Length: Tommee Tippee vs Competitors**")
    if "title_word_count" in tt_df.columns:
        tt_median_wc   = tt_df["title_word_count"].median()
        comp_median_wc = comp_df["title_word_count"].median() if len(comp_df) > 0 else None

        fig_tt_len = px.histogram(
            tt_df, x="title_word_count", nbins=20,
            title="Tommee Tippee Title Word Count Distribution",
            color_discrete_sequence=["#4A90D9"],
        )
        fig_tt_len.update_layout(xaxis_title="Word Count", yaxis_title="Number of Products")
        fig_tt_len.add_vline(
            x=tt_median_wc, line_dash="dash", line_color="blue",
            annotation_text=f"TT Median: {tt_median_wc:.0f}",
        )
        if comp_median_wc is not None:
            # Red dashed line shows where competitors sit — easy visual benchmark
            fig_tt_len.add_vline(
                x=comp_median_wc, line_dash="dot", line_color="red",
                annotation_text=f"Competitor Median: {comp_median_wc:.0f}",
            )
        st.plotly_chart(fig_tt_len, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("TT Median", f"{int(tt_median_wc)} words")
        c2.metric("TT Avg", f"{tt_df['title_word_count'].mean():.1f} words")
        if comp_median_wc is not None:
            c3.metric("Competitor Median", f"{int(comp_median_wc)} words")
    else:
        st.info("title_word_count column not found.")

    # ── 2c: Title quality score comparison ────────────────────────────────
    st.markdown("**Title Quality Score: Tommee Tippee vs Competitors**")
    st.caption(
        "The Title Quality Score measures how many key structural elements "
        "(product type, feature, material, size, quantity, age, flow rate) each title includes."
    )
    tt_title_scores   = compute_title_quality_scores(tt_df)
    comp_title_scores = compute_title_quality_scores(comp_df)
    if not tt_title_scores.empty and not comp_title_scores.empty:
        tt_tqs_avg   = tt_title_scores["title_quality_score"].mean()
        comp_tqs_avg = comp_title_scores["title_quality_score"].mean()
        delta_tqs    = tt_tqs_avg - comp_tqs_avg
        c1, c2, c3 = st.columns(3)
        c1.metric("TT Avg Title Quality", f"{tt_tqs_avg:.0f} / 100")
        c2.metric("Competitor Avg Title Quality", f"{comp_tqs_avg:.0f} / 100")
        c3.metric(
            "TT vs Competitor Gap",
            f"{delta_tqs:+.0f} pts",
            help="Positive = Tommee Tippee titles are more complete. Negative = competitors lead.",
        )

    # ── 2d: Top Tommee Tippee title examples ──────────────────────────────
    st.markdown("**Top Tommee Tippee Title Examples**")
    st.caption("Top 15 Tommee Tippee listings sorted by review count.")

    if "title" in tt_df.columns:
        ex_cols = [c for c in ["title", "brand", "reviewsCount", "stars", "price_value"] if c in tt_df.columns]
        tt_examples = (
            tt_df[ex_cols].dropna(subset=["title"])
            .sort_values("reviewsCount", ascending=False)
            .head(15).reset_index(drop=True)
        )
        col_cfg: dict = {}
        if "stars" in tt_examples.columns:
            col_cfg["stars"] = st.column_config.NumberColumn("Rating", format="%.1f \u2605")
        if "reviewsCount" in tt_examples.columns:
            col_cfg["reviewsCount"] = st.column_config.NumberColumn("Reviews", format="%d")
        if "price_value" in tt_examples.columns:
            col_cfg["price_value"] = st.column_config.NumberColumn("Price", format="$%.2f")
        st.dataframe(tt_examples, column_config=col_cfg, use_container_width=True, hide_index=True)

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3: DESCRIPTION INTELLIGENCE
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("3. Description Intelligence")

    if "description" not in tt_df.columns:
        st.info("No 'description' column found.")
    else:
        # ── 3a: Description keyword/theme frequency (TT only) ─────────────
        st.markdown("**Top Messaging Themes in Tommee Tippee Descriptions**")
        tt_desc_kw = extract_keyword_frequencies(
            tt_df["description"], DESC_PHRASE_MAP, DESC_STOPWORDS, top_n=25
        )
        if not tt_desc_kw.empty:
            fig_tt_desc = px.bar(
                tt_desc_kw, x="keyword", y="frequency",
                title="Most Frequent Keywords and Claims in Tommee Tippee Descriptions",
                color="frequency", color_continuous_scale="Greens",
            )
            fig_tt_desc.update_layout(
                showlegend=False,
                xaxis_title="Theme / Keyword", yaxis_title="Frequency", xaxis_tickangle=-35,
            )
            st.plotly_chart(fig_tt_desc, use_container_width=True)
            st.dataframe(
                tt_desc_kw.rename(columns={"keyword": "Theme / Keyword", "frequency": "Frequency"}),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No description keywords found for Tommee Tippee.")

        # ── 3b: Marketing claims frequency (TT only) ──────────────────────
        st.markdown("**Product Claims in Tommee Tippee Descriptions**")
        st.caption("Each listing counted once per claim, regardless of repetition.")

        tt_claims = build_claim_presence(tt_df, "description", CLAIM_MAP)
        tt_active_claims = tt_claims[tt_claims["listings"] > 0].copy()

        if not tt_active_claims.empty:
            fig_tt_claims = px.bar(
                tt_active_claims, x="claim", y="listings",
                title="Marketing Claims in Tommee Tippee Descriptions",
                color="listings", color_continuous_scale="Reds",
            )
            fig_tt_claims.update_layout(
                showlegend=False,
                xaxis_title="Claim", yaxis_title="Number of Listings", xaxis_tickangle=-20,
            )
            st.plotly_chart(fig_tt_claims, use_container_width=True)
            st.dataframe(
                tt_active_claims.rename(columns={"claim": "Marketing Claim", "listings": "Listings"}),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No marketing claims detected in Tommee Tippee descriptions.")

        # ── 3c: Description length distribution (TT vs competitor median) ──
        st.markdown("**Description Length: Tommee Tippee vs Competitors**")
        if "description_length" in tt_df.columns:
            tt_med_len   = tt_df["description_length"].median()
            comp_med_len = comp_df["description_length"].median() if len(comp_df) > 0 else None

            fig_tt_dlen = px.histogram(
                tt_df, x="description_length", nbins=20,
                title="Tommee Tippee Description Length Distribution",
                color_discrete_sequence=["#6B8E23"],
            )
            fig_tt_dlen.update_layout(xaxis_title="Character Count", yaxis_title="Number of Products")
            fig_tt_dlen.add_vline(
                x=tt_med_len, line_dash="dash", line_color="green",
                annotation_text=f"TT Median: {int(tt_med_len):,}",
            )
            if comp_med_len is not None:
                fig_tt_dlen.add_vline(
                    x=comp_med_len, line_dash="dot", line_color="red",
                    annotation_text=f"Competitor Median: {int(comp_med_len):,}",
                )
            st.plotly_chart(fig_tt_dlen, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("TT Median", f"{int(tt_med_len):,} chars")
            c2.metric("TT Avg", f"{int(tt_df['description_length'].mean()):,} chars")
            if comp_med_len is not None:
                c3.metric("Competitor Median", f"{int(comp_med_len):,} chars")
        else:
            st.info("description_length column not found.")

        # ── 3d: Description quality score comparison ───────────────────────
        st.markdown("**Description Quality Score: Tommee Tippee vs Competitors**")
        st.caption(
            "The Description Quality Score measures how many key content sections "
            "(Opening Claim, Feature Details, Benefit Language, Safety & Materials, "
            "Age & Fit, Care & Usage, Trust Signals) each description covers."
        )
        tt_desc_quality   = compute_description_quality_scores(tt_df)
        comp_desc_quality = compute_description_quality_scores(comp_df)
        if not tt_desc_quality.empty and not comp_desc_quality.empty:
            tt_dqs_avg   = tt_desc_quality["description_quality_score"].mean()
            comp_dqs_avg = comp_desc_quality["description_quality_score"].mean()
            delta_dqs    = tt_dqs_avg - comp_dqs_avg
            c1, c2, c3 = st.columns(3)
            c1.metric("TT Avg Description Quality",         f"{tt_dqs_avg:.0f} / 100")
            c2.metric("Competitor Avg Description Quality", f"{comp_dqs_avg:.0f} / 100")
            c3.metric(
                "TT vs Competitor Gap",
                f"{delta_dqs:+.0f} pts",
                help="Positive = Tommee Tippee descriptions are more complete. Negative = competitors lead.",
            )

        # ── 3e: Top Tommee Tippee description examples ────────────────────
        st.markdown("**Top Tommee Tippee Description Examples**")
        st.caption("Top 10 listings sorted by review count. Descriptions truncated for readability.")

        if "description" in tt_df.columns and "title" in tt_df.columns:
            desc_ex_cols = [
                c for c in ["title", "brand", "reviewsCount", "stars", "description"]
                if c in tt_df.columns
            ]
            desc_examples = (
                tt_df[desc_ex_cols].dropna(subset=["title"])
                .sort_values("reviewsCount", ascending=False)
                .head(10).reset_index(drop=True)
            )
            # Truncate description column to keep the table readable
            if "description" in desc_examples.columns:
                desc_examples = desc_examples.copy()
                desc_examples["description"] = desc_examples["description"].astype(str).str[:200] + "\u2026"

            desc_col_cfg: dict = {}
            if "stars" in desc_examples.columns:
                desc_col_cfg["stars"] = st.column_config.NumberColumn("Rating", format="%.1f \u2605")
            if "reviewsCount" in desc_examples.columns:
                desc_col_cfg["reviewsCount"] = st.column_config.NumberColumn("Reviews", format="%d")
            st.dataframe(
                desc_examples, column_config=desc_col_cfg,
                use_container_width=True, hide_index=True,
            )

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 4: COMPETITIVE GAP ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("4. Competitive Gap Analysis")
    st.caption(
        "Where competitors outpace Tommee Tippee on specific phrases and claims. "
        "Positive gap values (%) mean competitors use that keyword/claim more often — "
        "these represent the biggest content improvement opportunities. "
        "All frequencies are normalized by group size for a fair comparison."
    )

    if len(comp_df) == 0:
        st.info("No competitor products in the current filtered data to compare against.")
    else:
        # ── 4a: Title keyword gap ──────────────────────────────────────────
        st.markdown("**Title Phrase Gap: Tommee Tippee vs Competitors**")
        if title_col in tt_df.columns and title_col in comp_df.columns:
            title_gap_df = build_keyword_gap_table(
                tt_df, comp_df, title_col, TITLE_PHRASE_MAP, TITLE_STOPWORDS, top_n=40
            )
            # Show only rows where either side uses the keyword
            title_gap_display = title_gap_df[
                (title_gap_df["TT Count"] > 0) | (title_gap_df["Comp Count"] > 0)
            ].head(25)
            gap_col = "Gap % (Comp \u2212 TT)"
            st.dataframe(
                title_gap_display,
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("Title column not available for gap analysis.")
            title_gap_df = pd.DataFrame()

        # ── 4b: Description keyword/theme gap ─────────────────────────────
        st.markdown("**Description Messaging Theme Gap: Tommee Tippee vs Competitors**")
        if "description" in tt_df.columns and "description" in comp_df.columns:
            desc_gap_df = build_keyword_gap_table(
                tt_df, comp_df, "description", DESC_PHRASE_MAP, DESC_STOPWORDS, top_n=40
            )
            desc_gap_display = desc_gap_df[
                (desc_gap_df["TT Count"] > 0) | (desc_gap_df["Comp Count"] > 0)
            ].head(25)
            st.dataframe(
                desc_gap_display,
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("Description column not available for gap analysis.")
            desc_gap_df = pd.DataFrame()

        # ── 4c: Marketing claims gap ───────────────────────────────────────
        st.markdown("**Product Claims Gap: Tommee Tippee vs Competitors**")
        if "description" in tt_df.columns and "description" in comp_df.columns:
            claim_gap_df = build_claim_gap_table(tt_df, comp_df, "description", CLAIM_MAP)
            st.dataframe(
                claim_gap_df,
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("Description column not available for claims gap analysis.")
            claim_gap_df = pd.DataFrame()

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 5: IMAGE GALLERY
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("5. Tommee Tippee Image Gallery")
    st.caption(
        "Tommee Tippee product thumbnails sorted by review count descending. "
        "Review how the brand styles its hero images across listings."
    )

    img_col = "thumbnailImage" if "thumbnailImage" in tt_df.columns else None
    if img_col:
        tt_img_df = tt_df[
            tt_df[img_col].notna() & tt_df[img_col].astype(str).str.startswith("http")
        ].copy()

        if "reviewsCount" in tt_img_df.columns:
            tt_img_df = tt_img_df.sort_values("reviewsCount", ascending=False)

        if not tt_img_df.empty:
            max_tt_images = st.slider(
                "Max Tommee Tippee images to display",
                min_value=4, max_value=60, value=40, step=4,
                key="tt_max_images",
            )
            render_image_grid(tt_img_df.head(max_tt_images).reset_index(drop=True), img_col)
        else:
            st.info("No valid Tommee Tippee thumbnail images found.")
    else:
        st.info("thumbnailImage column not found.")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 6: RECOMMENDATIONS
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("6. Recommendations & Opportunities")
    st.caption(
        "Prioritized recommendations derived from the gap analysis above. "
        "Each insight identifies a specific content area where Tommee Tippee "
        "has room to close the gap with competitors."
    )

    # Only generate recommendations if we ran the gap analysis successfully
    if len(comp_df) == 0:
        st.info("No competitor data available — gap-based recommendations require at least one competitor product.")
    else:
        # Use empty DataFrames as fallbacks if a text column was missing
        _title_gap  = title_gap_df  if "title_gap_df"  in dir() and not title_gap_df.empty  else pd.DataFrame(columns=["Keyword / Phrase", "TT Count", "TT %", "Comp Count", "Comp %", f"Gap % (Comp \u2212 TT)"])
        _desc_gap   = desc_gap_df   if "desc_gap_df"   in dir() and not desc_gap_df.empty   else pd.DataFrame(columns=["Keyword / Phrase", "TT Count", "TT %", "Comp Count", "Comp %", f"Gap % (Comp \u2212 TT)"])
        _claim_gap  = claim_gap_df  if "claim_gap_df"  in dir() and not claim_gap_df.empty  else pd.DataFrame(columns=["claim", "TT Listings", "TT %", "Comp Listings", "Comp %", f"Gap % (Comp \u2212 TT)"])

        insights = generate_tt_recommendations(tt_df, comp_df, _title_gap, _desc_gap, _claim_gap)

        for insight in insights:
            st.markdown(f"- {insight}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.title("Amazon Competitive Listing Intelligence")
    st.caption(
        "Analyze how competitors craft their titles, descriptions, and images. "
        "Use the sidebar to filter by brand, category, price, or rating."
    )

    # Load data once per session (cached)
    df_raw = load_data(CLEANED_FILE)
    if df_raw.empty:
        st.error(
            "Cleaned data file not found or empty. "
            "Run `scripts/clean_data.py` first to generate the dataset."
        )
        return

    # Apply sidebar filters — produces the single filtered view used by all tabs
    df = apply_sidebar_filters(df_raw)

    if df.empty:
        st.warning(
            "No products match the current filters. "
            "Try widening your selection in the sidebar."
        )
        return

    # Six-tab layout — first five are the existing general dashboard;
    # sixth is the new Tommee Tippee Deep Dive.
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview",
        "Title Intelligence",
        "Description Intelligence",
        "Image Analysis",
        "Product Explorer",
        "Tommee Tippee Deep Dive",
    ])

    with tab1:
        tab_overview(df)
    with tab2:
        tab_title_intelligence(df)
    with tab3:
        tab_description_intelligence(df)
    with tab4:
        tab_image_analysis(df)
    with tab5:
        tab_product_explorer(df)
    with tab6:
        tab_tommee_tippee(df)


if __name__ == "__main__":
    main()
