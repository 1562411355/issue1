"""孤立森林：窗口展平后 iForest，-decision_function 作异常分数。"""
from __future__ import annotations

from sklearn.ensemble import IsolationForest

from . import BaseModel, register


@register
class IForest(BaseModel):
    name = "iforest"

    def __init__(self, n_estimators=100, contamination=0.02):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.clf = None

    def fit(self, train_w, val_w, patience=10, max_epochs=50, seed=0):
        super().fit(train_w, val_w, patience, max_epochs, seed)
        X = train_w.reshape(len(train_w), -1)
        self.clf = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=seed,
            n_jobs=-1,
        ).fit(X)
        return self

    def score_windows(self, W):
        return -self.clf.decision_function(W.reshape(len(W), -1))

    def count_params(self):
        return int(sum(t.tree_.node_count for t in self.clf.estimators_))
