"""Handcrafted feature extraction with Sastrawi stemmer.

Features are organized into groups for ablation studies:
    - length:    ans_wc, ans_cc, len_ratio
    - overlap:   kw_overlap_n, kw_overlap_r, kw_coverage
    - sastrawi:  root_overlap
    - diversity: jaccard, ttr
    - structure: bigram_ov, completeness
"""

import numpy as np
import pandas as pd
from typing import Optional

# Lazy-load Sastrawi stemmer (expensive initialization)
_stemmer = None
_sastrawi_available = None


def _get_stemmer():
    """Lazy-initialize Sastrawi stemmer."""
    global _stemmer, _sastrawi_available
    if _sastrawi_available is None:
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            _stemmer = StemmerFactory().create_stemmer()
            _sastrawi_available = True
        except ImportError:
            _sastrawi_available = False
            print("[WARNING] Sastrawi not installed. root_overlap will be 0.")
    return _stemmer


# --- Feature Group Registry ---

FEATURE_GROUPS = {
    "length":    ["ans_wc", "ans_cc", "len_ratio"],
    "overlap":   ["kw_overlap_n", "kw_overlap_r", "kw_coverage"],
    "sastrawi":  ["root_overlap"],
    "diversity": ["jaccard", "ttr"],
    "structure": ["bigram_ov", "completeness"],
}

ALL_FEATURES = []
for group_features in FEATURE_GROUPS.values():
    ALL_FEATURES.extend(group_features)


def get_feature_names(groups: Optional[list[str]] = None) -> list[str]:
    """Get feature names for specified groups (or all if None).
    
    Args:
        groups: List of group names (e.g., ["overlap", "sastrawi"]).
                If None, returns all 12 features.
                
    Returns:
        List of feature column names.
    """
    if groups is None:
        return ALL_FEATURES.copy()
    names = []
    for g in groups:
        if g not in FEATURE_GROUPS:
            raise ValueError(f"Unknown group '{g}'. Available: {list(FEATURE_GROUPS.keys())}")
        names.extend(FEATURE_GROUPS[g])
    return names


def _extract_row(ans: str, ref: str) -> dict:
    """Extract all 12 handcrafted features from a single (answer, reference) pair.
    
    Args:
        ans: Student answer text.
        ref: Reference answer text.
        
    Returns:
        Dictionary of feature name → value.
    """
    ans = str(ans).lower()
    ref = str(ref).lower()
    aw, rw = ans.split(), ref.split()
    a_set, r_set = set(aw), set(rw)
    overlap = a_set & r_set

    f = {}
    # Length features
    f["ans_wc"] = len(aw)
    f["ans_cc"] = len(ans)
    f["len_ratio"] = len(aw) / max(len(rw), 1)
    
    # Overlap features
    f["kw_overlap_n"] = len(overlap)
    f["kw_overlap_r"] = len(overlap) / max(len(r_set), 1)
    f["kw_coverage"] = len(overlap) / max(len(a_set), 1)
    
    # Sastrawi root overlap
    stemmer = _get_stemmer()
    if stemmer is not None:
        a_stem = set(stemmer.stem(w) for w in aw)
        r_stem = set(stemmer.stem(w) for w in rw)
        f["root_overlap"] = len(a_stem & r_stem) / max(len(r_stem), 1)
    else:
        f["root_overlap"] = 0.0
    
    # Diversity features
    union = a_set | r_set
    f["jaccard"] = len(overlap) / max(len(union), 1)
    f["ttr"] = len(a_set) / max(len(aw), 1)
    
    # Structure features
    ab = set(zip(aw[:-1], aw[1:])) if len(aw) > 1 else set()
    rb = set(zip(rw[:-1], rw[1:])) if len(rw) > 1 else set()
    f["bigram_ov"] = len(ab & rb) / max(len(rb), 1)
    
    chunks = ref.split(",")
    cov = sum(1 for c in chunks if any(w in ans for w in c.split() if len(w) > 3))
    f["completeness"] = cov / max(len(chunks), 1)
    
    return f


def extract_features(df: pd.DataFrame,
                     groups: Optional[list[str]] = None,
                     ans_col: str = "student_answer",
                     ref_col: str = "reference_answer") -> np.ndarray:
    """Extract handcrafted features from a DataFrame.
    
    Args:
        df: DataFrame with answer and reference columns.
        groups: Feature groups to include (None = all 12 features).
        ans_col: Column name for student answers.
        ref_col: Column name for reference answers.
        
    Returns:
        numpy array of shape (n_samples, n_features).
    """
    target_names = get_feature_names(groups)
    
    print(f"[Features] Extracting {len(target_names)} HC features...")
    rows = []
    for _, row in df.iterrows():
        rows.append(_extract_row(row[ans_col], row[ref_col]))
    
    feat_df = pd.DataFrame(rows)
    return feat_df[target_names].values


def extract_features_df(df: pd.DataFrame,
                        groups: Optional[list[str]] = None,
                        ans_col: str = "student_answer",
                        ref_col: str = "reference_answer") -> pd.DataFrame:
    """Same as extract_features but returns a DataFrame with named columns.
    
    Args:
        df: DataFrame with answer and reference columns.
        groups: Feature groups to include (None = all).
        ans_col: Column name for student answers.
        ref_col: Column name for reference answers.
        
    Returns:
        DataFrame with feature columns.
    """
    target_names = get_feature_names(groups)
    X = extract_features(df, groups, ans_col, ref_col)
    return pd.DataFrame(X, columns=target_names, index=df.index)
