"""Tree-model backends for TMB regression (XGBoost, LightGBM, scikit-learn)."""

from __future__ import annotations

TREE_BACKENDS = ("xgboost", "lightgbm", "random_forest", "gradient_boosting")


def make_regressor(backend: str, params: dict | None = None):
    """Build a fitted-ready regressor for ``backend`` with ``params``."""
    if backend not in TREE_BACKENDS:
        raise ValueError(f"backend must be one of {TREE_BACKENDS}; got {backend!r}.")
    params = dict(params or {})
    if backend == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("XGBoost is a core dependency. Reinstall with: pip install -e .") from exc
        params.setdefault("objective", "reg:squarederror")
        return xgb.XGBRegressor(**params)
    if backend == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError('LightGBM backend requires: pip install -e ".[lightgbm]"') from exc
        return lgb.LGBMRegressor(**params)
    if backend == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**params)
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(**params)
