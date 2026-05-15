"""Late Fusion via Ridge Regression stacking."""

import numpy as np
from sklearn.linear_model import Ridge


class LateFusion:
    """Prediction-level fusion of Deep and Shallow model predictions.
    
    Uses Ridge Regression to learn optimal weights for combining
    predictions from IndoBERT and HC Sastrawi models.
    
    The learned weights provide architectural interpretability:
        Final_Score = w_deep * IndoBERT_pred + w_shallow * HC_pred + bias
    """
    
    def __init__(self, alpha: float = 1.0):
        """Initialize Late Fusion stacker.
        
        Args:
            alpha: Ridge regularization strength.
        """
        self.ridge = Ridge(alpha=alpha)
        self._fitted = False
    
    def fit(self, pred_deep: np.ndarray, pred_shallow: np.ndarray,
            y_true: np.ndarray) -> "LateFusion":
        """Fit stacking model on OOF predictions.
        
        Args:
            pred_deep: OOF predictions from deep model (IndoBERT).
            pred_shallow: OOF predictions from shallow model (HC+SVR).
            y_true: Ground truth scores.
            
        Returns:
            self (for chaining).
        """
        X = np.column_stack([pred_deep, pred_shallow])
        self.ridge.fit(X, y_true)
        self._fitted = True
        return self
    
    def predict(self, pred_deep: np.ndarray,
                pred_shallow: np.ndarray) -> np.ndarray:
        """Generate fused predictions.
        
        Args:
            pred_deep: Predictions from deep model.
            pred_shallow: Predictions from shallow model.
            
        Returns:
            Fused predictions clipped to [0, 1].
        """
        X = np.column_stack([pred_deep, pred_shallow])
        return np.clip(self.ridge.predict(X), 0, 1)
    
    def get_weights(self) -> dict:
        """Extract interpretable weights from the stacker.
        
        Returns:
            Dictionary with:
                - w_deep: weight for deep model
                - w_shallow: weight for shallow model
                - bias: intercept
                - pct_deep: percentage contribution of deep model
                - pct_shallow: percentage contribution of shallow model
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        
        w = self.ridge.coef_
        total = abs(w[0]) + abs(w[1])
        
        return {
            "w_deep": w[0],
            "w_shallow": w[1],
            "bias": self.ridge.intercept_,
            "pct_deep": abs(w[0]) / total if total > 0 else 0.5,
            "pct_shallow": abs(w[1]) / total if total > 0 else 0.5,
        }
