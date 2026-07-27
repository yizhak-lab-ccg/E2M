from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def make_folds(cancer: pd.Series, n_splits: int, random_state: int):
    if cancer.nunique() > 1 and cancer.value_counts().min() >= n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return splitter.split(np.arange(len(cancer)), cancer), "stratified_by_cancer"
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return splitter.split(np.arange(len(cancer))), "shuffled_kfold"
