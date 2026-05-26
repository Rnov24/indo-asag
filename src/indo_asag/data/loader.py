"""Data loading and preprocessing for ASAG Indonesia dataset."""

import pandas as pd
import numpy as np
from pathlib import Path


def load_dataset(path: str, score_col: str = "score_manual_avg",
                 score_max: float = 100.0) -> pd.DataFrame:
    """Load the ASAG dataset from parquet and apply standard preprocessing.
    
    Preprocessing steps:
        1. Drop rows with missing scores, student_answer, or reference_answer.
        2. Drop rows with empty string answers.
        3. Normalize scores to [0, 1] range.
        4. Add stratification bins for balanced k-fold splitting.
    
    Args:
        path: Path to the parquet file.
        score_col: Name of the score column.
        score_max: Maximum score value for normalization.
        
    Returns:
        Cleaned DataFrame with added `score_norm` and `score_bin` columns.
    """
    df = pd.read_parquet(path)
    n_raw = len(df)
    
    # Clean: drop missing and empty
    required_cols = [score_col, "student_answer", "reference_answer"]
    df = df.dropna(subset=required_cols)
    df = df[df["student_answer"].str.strip() != ""]
    df = df[df["reference_answer"].str.strip() != ""]
    df = df.reset_index(drop=True)
    
    # Normalize score to [0, 1]
    df["score_norm"] = df[score_col] / score_max
    
    # Stratification bins for k-fold
    df["score_bin"] = pd.qcut(df["score_norm"], q=5, labels=False, duplicates="drop")
    
    print(f"[Data] Loaded {n_raw} -> {len(df)} rows (cleaned)")
    print(f"[Data] Questions: {df['question_id'].nunique()}, Topics: {df['topic'].nunique()}")
    
    return df
