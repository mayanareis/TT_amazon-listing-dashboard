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
    # Brand-name fragments — individual words that leak out when a brand name
    # is not caught by the phrase map (e.g. if spacing/punctuation varies).
    # The full brand tokens (tommee_tippee, dr_browns) are kept via phrase map.
    "tommee", "tippee", "brown", "dr",
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


# ══════════════════════════════════════════════════════════════════════════════
# MESSAGING CATEGORIES
# Used by the Description Messaging Structure analysis.
# Each category maps to a list of regex patterns; a listing "covers" the
# category if ANY pattern matches its description (binary, not frequency).
# ══════════════════════════════════════════════════════════════════════════════
MESSAGING_CATEGORIES: dict = {
    "Social Proof": [
        r"\b#\s*1\s+brand\b",
        r"\brecommended\s+by\s+(moms?|doctors?|pediatricians?)\b",
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

# Specific social proof phrases for fine-grained detection within the
# "Social Proof" category above.
SOCIAL_PROOF_PATTERNS: dict = {
    "#1 Brand":                  r"\b#\s*1\s+brand\b",
    "Recommended by Moms":       r"\brecommended\s+by\s+moms?\b",
    "Recommended by Doctors":    r"\brecommended\s+by\s+doctors?\b",
    "Clinically Tested":         r"\bclinically\s+tested\b",
    "Survey / Study":            r"\bsurvey\b|\bstud(?:y|ied)\b",
    "Award Winning":             r"\baward[- ]?winning\b",
    "Trusted By":                r"\btrusted\s+by\b",
    "Millions of Moms":          r"\bmillion[s]?\s+(of\s+)?moms?\b",
    "Pediatrician Recommended":  r"\bpediatrician\b",
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
                # Step 3a — phrase token: always keep (no stopword check)
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
    """Render the Overview tab: KPI metrics and brand distribution charts."""
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

    st.divider()

    # ── Brand distribution
    st.subheader("Brand Distribution")
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
            title="Top 15 Brands by Product Count",
            color="products", color_continuous_scale="Blues",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Products", xaxis_tickangle=-30)
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
    """
    st.header("Title Intelligence")
    st.caption(
        "Uncover patterns in how competitors write product titles — "
        "keyword choices, phrase usage, length strategy, and top-performing examples."
    )

    # Prefer the lowercased clean_title for cleaner tokenization
    title_col = "clean_title" if "clean_title" in df.columns else "title"

    # ── Section 1: Keyword / phrase frequency ─────────────────────────────────
    st.subheader("Top Keywords and Phrases in Competitor Titles")
    st.caption(
        "Multi-word phrases like 'Anti Colic' or 'BPA Free' are grouped as single concepts "
        "before counting, so they appear as meaningful units rather than split individual words."
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
                title="Most Frequent Keywords and Phrases in Competitor Titles",
                color="frequency",
                color_continuous_scale="Teal",
            )
            fig_kw.update_layout(
                showlegend=False,
                xaxis_title="Keyword / Phrase",
                yaxis_title="Frequency",
                xaxis_tickangle=-35,
            )
            st.plotly_chart(fig_kw, use_container_width=True)

            # Small companion table for exact counts
            st.dataframe(
                kw_df.rename(columns={"keyword": "Keyword / Phrase", "frequency": "Frequency"}),
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
    st.subheader("Top Listing Title Examples")
    st.caption(
        "Top 15 competitor listings sorted by review count — "
        "these are the titles that have driven the most customer engagement."
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
        # Configure column display labels and formats
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB: DESCRIPTION INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

def tab_description_intelligence(df: pd.DataFrame) -> None:
    """Render the Description Intelligence tab.

    Sections:
        1. Top keywords / themes — phrase-aware bar chart + table
        2. Marketing claims frequency — listing-level presence detection
        3. Description length distribution
    """
    st.header("Description Intelligence")
    st.caption(
        "What language and persuasion tactics are competitors using? "
        "Find the themes, claims, and length patterns that define the category."
    )

    if "description" not in df.columns:
        st.error("No 'description' column found. Re-run scripts/clean_data.py.")
        return

    # ── Section 1: Theme / keyword extraction ─────────────────────────────────
    st.subheader("Most Frequent Keywords and Claims in Descriptions")
    st.caption(
        "Multi-word phrases are preserved as single concepts before counting. "
        "Generic filler words are excluded to surface meaningful signals."
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
            title="Most Frequent Keywords and Claims in Descriptions",
            color="frequency", color_continuous_scale="Greens",
        )
        fig_themes.update_layout(
            showlegend=False,
            xaxis_title="Theme / Keyword",
            yaxis_title="Frequency",
            xaxis_tickangle=-35,
        )
        st.plotly_chart(fig_themes, use_container_width=True)

        # Companion table
        st.dataframe(
            desc_kw_df.rename(columns={"keyword": "Theme / Keyword", "frequency": "Frequency"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No description keywords found after filtering stopwords.")

    st.divider()

    # ── Section 2: Marketing claims — listing-level presence ──────────────────
    st.subheader("Marketing Claims Frequency")
    st.caption(
        "How many distinct listings include each marketing claim in their description? "
        "Each listing is counted once regardless of how many times a claim appears."
    )

    claims_df = build_claim_presence(df, "description", CLAIM_MAP)
    active_claims = claims_df[claims_df["listings"] > 0].copy()

    if not active_claims.empty:
        fig_claims = px.bar(
            active_claims, x="claim", y="listings",
            title="Marketing Claims Frequency (Listings Containing Each Claim)",
            color="listings", color_continuous_scale="Reds",
        )
        fig_claims.update_layout(
            showlegend=False,
            xaxis_title="Claim",
            yaxis_title="Number of Listings",
            xaxis_tickangle=-20,
        )
        st.plotly_chart(fig_claims, use_container_width=True)

        # Table with percentage column for client-facing reporting
        st.dataframe(
            active_claims.rename(columns={"claim": "Marketing Claim", "listings": "Listings"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No marketing claims detected across descriptions.")

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

    # ── Section 4: Description Messaging Structure ─────────────────────────────
    st.subheader("Description Messaging Structure")
    st.caption(
        "How descriptions are constructed across six strategic messaging categories. "
        "Each listing is scored binary per category — present if any matching phrase "
        "appears, absent otherwise."
    )

    # ── 4a: Social proof signals ───────────────────────────────────────────────
    st.markdown("**Social Proof Signals**")
    st.caption(
        "How often do listings use specific social proof phrases "
        "such as '#1 brand', 'recommended by moms', or 'clinically tested'?"
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
    st.markdown("**Messaging Category Coverage (All Listings)**")
    st.caption(
        "What percentage of listings include language from each messaging category? "
        "A category is marked present if any of its patterns match in the description."
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

    # ── 4c: Claim coverage score distribution ──────────────────────────────────
    st.markdown("**Claim Coverage Score Distribution**")
    st.caption(
        "Each listing receives a score from 0–100 based on how many of the six "
        "messaging categories appear in its description. "
        "100 = all six categories covered."
    )
    scores_df = build_messaging_category_scores(df, "description")
    avg_score = scores_df["claim_coverage_score"].mean()

    fig_score = px.histogram(
        scores_df, x="claim_coverage_score", nbins=20,
        title="Claim Coverage Score Distribution",
        color_discrete_sequence=["#7B68EE"],
    )
    fig_score.update_layout(
        xaxis_title="Coverage Score (0–100)", yaxis_title="Number of Listings",
    )
    fig_score.add_vline(
        x=avg_score, line_dash="dash", line_color="red",
        annotation_text=f"Avg: {avg_score:.1f}",
    )
    st.plotly_chart(fig_score, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Avg Coverage Score", f"{avg_score:.1f} / 100")
    fully_covered = int((scores_df["claim_coverage_score"] == 100).sum())
    c2.metric("Listings Covering All 6 Categories", f"{fully_covered:,}")

    st.divider()

    # ── Section 5: Tommee Tippee Messaging Opportunities ──────────────────────
    st.subheader("Tommee Tippee Messaging Opportunities")
    st.caption(
        "Tommee Tippee's description messaging compared against all other competitors "
        "in the filtered dataset. Taller competitor bars highlight categories where "
        "Tommee Tippee's content lags — these are the clearest improvement opportunities."
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
        "compared directly against all other competitors in the filtered dataset."
    )

    # ── Identify Tommee Tippee products ───────────────────────────────────────
    # Case-insensitive match on any brand value containing "tommee".
    # This captures "Tommee Tippee", "TOMMEE TIPPEE", etc.
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
    st.subheader("1. Tommee Tippee Overview")

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
    # SECTION 2: TITLE ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("2. Title Analysis")

    # ── 2a: Title keyword frequency (TT only) ─────────────────────────────
    st.markdown("**Title Keyword Frequency (Tommee Tippee only)**")
    st.caption(
        "Multi-word phrases are grouped as single concepts using the same "
        "phrase logic applied across the rest of the dashboard."
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
    st.markdown("**Title Length Distribution**")
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

    # ── 2c: Top Tommee Tippee title examples ──────────────────────────────
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
    # SECTION 3: DESCRIPTION ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("3. Description Analysis")

    if "description" not in tt_df.columns:
        st.info("No 'description' column found.")
    else:
        # ── 3a: Description keyword/theme frequency (TT only) ─────────────
        st.markdown("**Description Keyword / Theme Frequency (Tommee Tippee only)**")
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
        st.markdown("**Marketing Claims Frequency (Tommee Tippee only)**")
        st.caption("Each listing counted once per claim, regardless of how many times it appears.")

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
        st.markdown("**Description Length Distribution**")
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

        # ── 3d: Top Tommee Tippee description examples ────────────────────
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
    # SECTION 4: COMPETITOR GAP ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("4. Competitor Gap Analysis")
    st.caption(
        "Positive gap values (in percentage points) mean competitors use that keyword/claim "
        "more often than Tommee Tippee — these are the biggest content opportunities. "
        "Frequencies are normalised by group size for a fair comparison."
    )

    if len(comp_df) == 0:
        st.info("No competitor products in the current filtered data to compare against.")
    else:
        # ── 4a: Title keyword gap ──────────────────────────────────────────
        st.markdown("**Title Keyword Gap: Tommee Tippee vs Competitors**")
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
        st.markdown("**Description Theme Gap: Tommee Tippee vs Competitors**")
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
        st.markdown("**Marketing Claims Gap: Tommee Tippee vs Competitors**")
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
    st.subheader("6. How Tommee Tippee Can Improve")
    st.caption(
        "Rule-based insights derived from the gap analysis above. "
        "Positive gaps indicate areas where competitors have stronger content coverage."
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
