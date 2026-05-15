"""Evaluation metrics for ASAG: QWK, Pearson, RMSE, MAE."""

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray,
                              n_classes: int = 5) -> float:
    """Compute Quadratic Weighted Kappa (gold standard for AES/ASAG).
    
    Scores are binned into `n_classes` categories before computing.
    
    Args:
        y_true: Ground truth scores in [0, 1].
        y_pred: Predicted scores in [0, 1].
        n_classes: Number of bins for discretization.
        
    Returns:
        QWK score (float). 1.0 = perfect, 0.0 = random, <0 = worse than random.
    """
    bins = np.linspace(0, 1, n_classes + 1)
    yt = np.digitize(y_true, bins[1:-1])
    yp = np.digitize(y_pred, bins[1:-1])
    n = len(yt)
    
    W = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)
    
    O = np.zeros((n_classes, n_classes))
    for t, p in zip(yt, yp):
        O[t, p] += 1
    O /= n
    
    E = np.outer(
        np.bincount(yt, minlength=n_classes),
        np.bincount(yp, minlength=n_classes)
    ).astype(float) / (n * n)
    
    den = np.sum(W * E)
    return 1.0 - np.sum(W * O) / den if den != 0 else 0.0


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all standard ASAG metrics.
    
    Args:
        y_true: Ground truth scores in [0, 1].
        y_pred: Predicted scores in [0, 1].
        
    Returns:
        Dictionary with keys: QWK, Pearson, RMSE, MAE.
    """
    return {
        "QWK": quadratic_weighted_kappa(y_true, y_pred),
        "Pearson": pearsonr(y_true, y_pred)[0],
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }
