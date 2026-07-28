"""Tree ML models (XGBoost, LightGBM, scikit-learn).

``make_regressor`` builds the TMB regressor; ``make_classifier`` builds the tree
classifier that Tree SHAP explains. Both share the same ``TREE_MODELS`` names and
per-model parameter blocks.
"""

from __future__ import annotations

TREE_MODELS = ("xgboost", "lightgbm", "random_forest", "gradient_boosting")


def make_regressor(ml_model: str, params: dict | None = None):
    """Build a fitted-ready regressor for ``ml_model`` with ``params``."""
    if ml_model not in TREE_MODELS:
        raise ValueError(f"ml_model must be one of {TREE_MODELS}; got {ml_model!r}.")
    params = dict(params or {})
    if ml_model == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("XGBoost is a core dependency. Reinstall with: pip install -e .") from exc
        params.setdefault("objective", "reg:squarederror")
        return xgb.XGBRegressor(**params)
    if ml_model == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError('LightGBM requires: pip install -e ".[lightgbm]"') from exc
        return lgb.LGBMRegressor(**params)
    if ml_model == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**params)
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(**params)


def make_classifier(ml_model: str, params: dict | None = None):
    """Build a tree classifier for ``ml_model`` that ``shap.TreeExplainer`` can explain."""
    if ml_model not in TREE_MODELS:
        raise ValueError(f"ml_model must be one of {TREE_MODELS}; got {ml_model!r}.")
    params = dict(params or {})
    if ml_model == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("XGBoost is a core dependency. Reinstall with: pip install -e .") from exc
        return xgb.XGBClassifier(**params)
    if ml_model == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError('LightGBM requires: pip install -e ".[lightgbm]"') from exc
        return lgb.LGBMClassifier(**params)
    if ml_model == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**params)
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(**params)
