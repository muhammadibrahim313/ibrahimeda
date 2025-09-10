from __future__ import annotations
import pandas as pd

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN = True
except Exception:
    SKLEARN = False

def quick_fit(model, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Fit a sklearn-like model and return fitted model and accuracy on a holdout."""
    if not SKLEARN:
        raise ImportError("Install scikit-learn: pip install 'ibrahimeda[ml]'")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    return model, round(acc,4)
