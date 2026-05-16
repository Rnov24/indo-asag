"""Cross-validation runners: single-seed, multi-seed, and LOQO."""

import numpy as np
import pandas as pd
from typing import Callable
from sklearn.model_selection import StratifiedKFold

from indo_asag.evaluation.metrics import evaluate


def run_cv(df: pd.DataFrame, predict_fn: Callable,
           n_folds: int = 5, seed: int = 42) -> tuple[np.ndarray, dict]:
    """Run stratified k-fold cross-validation.
    
    Args:
        df: DataFrame with `score_bin` column for stratification.
        predict_fn: Function(train_df, val_df, fold_idx) → predictions array.
        n_folds: Number of folds.
        seed: Random seed for fold splitting.
        
    Returns:
        Tuple of (oof_predictions, metrics_dict).
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    preds = np.zeros(len(df))
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, df["score_bin"])):
        import inspect
        sig = inspect.signature(predict_fn)
        if "seed" in sig.parameters:
            p = predict_fn(df.iloc[tr_idx], df.iloc[va_idx], fold, seed=seed)
        else:
            p = predict_fn(df.iloc[tr_idx], df.iloc[va_idx], fold)
        preds[va_idx] = np.clip(p, 0, 1)
    
    return preds, evaluate(df["score_norm"].values, preds)


def run_multi_seed(df: pd.DataFrame, predict_fn: Callable,
                   seeds: list[int] = None,
                   n_folds: int = 5) -> dict:
    """Run CV across multiple seeds for statistical validity.
    
    Args:
        df: DataFrame with `score_bin` column.
        predict_fn: Function(train_df, val_df, fold_idx) → predictions array.
        seeds: List of random seeds. Defaults to [42, 123, 456, 789, 1024].
        n_folds: Number of folds per seed.
        
    Returns:
        Dictionary with:
            - Per-metric: "mean ± std" strings
            - "_mean": dict of mean values
            - "_std": dict of std values
            - "_per_seed": list of per-seed metric dicts
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]
    
    results = []
    all_preds = {}
    
    for s in seeds:
        preds, metrics = run_cv(df, predict_fn, n_folds=n_folds, seed=s)
        all_preds[s] = preds
        results.append(metrics)
    
    mdf = pd.DataFrame(results)
    summary = {
        col: f"{mdf[col].mean():.4f} +/- {mdf[col].std():.4f}"
        for col in mdf.columns
    }
    summary["_mean"] = {col: mdf[col].mean() for col in mdf.columns}
    summary["_std"] = {col: mdf[col].std() for col in mdf.columns}
    summary["_per_seed"] = results
    summary["_preds"] = all_preds
    
    return summary


def run_loqo(df: pd.DataFrame, predict_fn: Callable,
             question_col: str = "question_id") -> tuple:
    """Leave-One-Question-Out evaluation for cross-prompt generalization.
    
    Args:
        df: DataFrame with question_id column.
        predict_fn: Function(train_df, val_df) → predictions array.
                    Note: no fold parameter (entire question is test set).
        question_col: Column name for question identifiers.
        
    Returns:
        Tuple of (loqo_metrics_df, full_predictions_array).
        loqo_metrics_df: DataFrame with per-question metrics.
        full_predictions_array: np.ndarray of shape (len(df),) with
            out-of-question predictions for every sample.
    """
    questions = df[question_col].unique()
    results = []
    full_preds = np.zeros(len(df))
    
    for q in sorted(questions):
        test_mask = df[question_col] == q
        train_df = df[~test_mask].reset_index(drop=True)
        test_df = df[test_mask].reset_index(drop=True)
        
        if len(test_df) < 5:
            continue
        
        preds = predict_fn(train_df, test_df)
        preds = np.clip(preds, 0, 1)
        
        # Store predictions at original indices
        full_preds[test_mask.values] = preds
        
        metrics = evaluate(test_df["score_norm"].values, preds)
        metrics["question_id"] = q
        metrics["n_test"] = len(test_df)
        results.append(metrics)
        
        print(f"  Q={q}: QWK={metrics['QWK']:.3f}, n={len(test_df)}")
    
    loqo_df = pd.DataFrame(results)
    mean_qwk = loqo_df["QWK"].mean()
    std_qwk = loqo_df["QWK"].std()
    print(f"\n  LOQO Mean QWK: {mean_qwk:.4f} +/- {std_qwk:.4f}")
    
    return loqo_df, full_preds
