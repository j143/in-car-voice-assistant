"""SVM-based intent classifier with OOD detection (optional).

This module is optional and only used when explicitly selected.
It lazy-loads sklearn to avoid heavy dependencies for minimal runs.
"""

from __future__ import annotations

from typing import Tuple
import logging

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

logger = logging.getLogger(__name__)


class SVMCommandClassifier:
    """Intent classifier with out-of-distribution detection using SVM.

    API:
      - fit(X, y)
      - predict(embeddings) -> (intent_label, confidence)
    """

    def __init__(
        self,
        svm_kernel: str = "rbf",
        ood_threshold: float = 0.3,
        n_intents: int = 5,
    ) -> None:
        self.svm_kernel = svm_kernel
        self.ood_threshold = ood_threshold
        self.n_intents = n_intents
        self._scaler = None
        self._svm = None
        self.is_fitted = False
        self.intent_labels = {
            0: "music_control",
            1: "navigation",
            2: "climate_control",
            3: "error_handling",
            4: "ood",
        }

    def _lazy_import(self):
        from sklearn.svm import SVC  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
        return SVC, StandardScaler

    def fit(self, X, y) -> None:
        if np is None:
            raise RuntimeError("numpy not available; cannot fit SVM classifier")
        SVC, StandardScaler = self._lazy_import()
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._svm = SVC(kernel=self.svm_kernel, probability=True, C=1.0)
        self._svm.fit(X_scaled, y)
        self.is_fitted = True
        logger.info("SVM classifier fitted successfully")

    def predict(self, embeddings) -> Tuple[str, float]:
        if not self.is_fitted or self._svm is None or self._scaler is None:
            return ("ood", 0.0)
        if np is None:
            return ("ood", 0.0)
        try:
            mean_emb = embeddings.mean(axis=0, keepdims=True)
            mean_scaled = self._scaler.transform(mean_emb)
            pred = self._svm.predict(mean_scaled)[0]
            proba = self._svm.predict_proba(mean_scaled).max()
            if float(proba) < self.ood_threshold:
                return ("ood", float(proba))
            return (self.intent_labels.get(int(pred), "ood"), float(proba))
        except Exception as e:  # pragma: no cover
            logger.error("SVM predict failed: %s", e)
            return ("ood", 0.0)
