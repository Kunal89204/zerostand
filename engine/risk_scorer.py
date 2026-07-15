"""
Request-stage (pre-approval) risk model. This is the layer that decides
whether a token gets minted at all -- prevention, not detection. Fuses:
declared scope-vs-role mismatch (deterministic flag from the simulator/real
entitlement catalog), the peer-group deviation, the self deviation, and the
*gap* between them (peer_deviation - self_deviation is the direct signal for
"consistent with themselves, inconsistent with their peers" -- the slow-drift
case a self-only baseline would miss).
"""
import math
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from datetime import datetime
from engine.peer_model import PeerModel

try:
    import shap
    SHAP_AVAILABLE = True
except Exception as exc:  # ImportError, DLL policy blocks, etc.
    shap = None
    SHAP_AVAILABLE = False
    warnings.warn(
        f"SHAP unavailable ({exc}); using RandomForest feature importances for explanations.",
        RuntimeWarning,
        stacklevel=1,
    )

CRITICALITY_NUM = {"low": 0.2, "medium": 0.5, "high": 0.9}

FEATURE_ORDER = ["scope_overreach", "criticality_num", "peer_deviation", "self_deviation",
                  "peer_self_gap", "hour_sin", "hour_cos"]


def request_features(case, peer_model: PeerModel):
    peer_dev = peer_model.peer_deviation(case["current_vector"], case["role"])
    self_dev = peer_model.self_deviation(case["current_vector"], case["own_recent_vector"])
    hour = datetime.fromisoformat(case["timestamp"]).hour
    return {
        "scope_overreach": float(case["scope_overreach"]),
        "criticality_num": CRITICALITY_NUM[case["criticality"]],
        "peer_deviation": peer_dev,
        "self_deviation": self_dev,
        "peer_self_gap": peer_dev - self_dev,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
    }


def to_vector(feats):
    return [feats[k] for k in FEATURE_ORDER]


class RiskScorer:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=23)
        self.peer_model = PeerModel()
        self.explainer = None
        self.metrics = {}

    def fit(self, cases):
        train, test = train_test_split(cases, test_size=0.25, random_state=23,
                                        stratify=[c["request_label"] for c in cases])
        self.peer_model.fit(train)

        X_train = np.array([to_vector(request_features(c, self.peer_model)) for c in train])
        y_train = np.array([1 if c["request_label"] == "anomalous" else 0 for c in train])
        self.rf.fit(X_train, y_train)
        self.explainer = shap.TreeExplainer(self.rf) if SHAP_AVAILABLE else None

        X_test = np.array([to_vector(request_features(c, self.peer_model)) for c in test])
        y_test = np.array([1 if c["request_label"] == "anomalous" else 0 for c in test])
        y_pred = self.rf.predict(X_test)

        tp = int(((y_pred == 1) & (y_test == 1)).sum())
        fp = int(((y_pred == 1) & (y_test == 0)).sum())
        fn = int(((y_pred == 0) & (y_test == 1)).sum())
        tn = int(((y_pred == 0) & (y_test == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        self.metrics = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                         "precision": round(precision, 3), "recall": round(recall, 3),
                         "test_size": len(test)}
        return self.metrics

    def score(self, case):
        feats = request_features(case, self.peer_model)
        x = np.array([to_vector(feats)])
        prob = float(self.rf.predict_proba(x)[0][1])

        if self.explainer is not None:
            shap_vals = self.explainer.shap_values(x)
            contribs = (
                shap_vals[0][:, 1]
                if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3
                else shap_vals[1][0]
            )
        else:
            # Fallback when SHAP/numba cannot load (e.g. Windows Application Control).
            contribs = self.rf.feature_importances_ * x[0]

        top_idx = np.argsort(-np.abs(contribs))[:4]
        reasons = [{"feature": FEATURE_ORDER[i], "value": round(float(feats[FEATURE_ORDER[i]]), 3),
                    "impact": round(float(contribs[i]), 3)} for i in top_idx]

        return {"risk_score": round(prob, 3), "features": feats, "top_reasons": reasons}
