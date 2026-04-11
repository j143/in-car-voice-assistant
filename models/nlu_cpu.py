"""Lightweight CPU-only NLU module.

Uses TF-IDF vectorisation + cosine similarity for intent classification.
No torch or GPU required.  Trains in <1 s from the automotive JSONL data.

Supported intents (from automotive_domain_train.jsonl):
  climate_control, error_handling, music_control, navigation,
  battery_warning, query_sensor, query_historical, set_alert, ood
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard-coded fallback training examples (used when JSONL files are absent)
# ---------------------------------------------------------------------------
_FALLBACK_EXAMPLES: List[Tuple[str, str]] = [
    ("set temperature to 22 degrees", "climate_control"),
    ("increase fan speed", "climate_control"),
    ("turn on AC", "climate_control"),
    ("cool down the cabin", "climate_control"),
    ("heated seats on", "climate_control"),
    ("defrost the windshield", "climate_control"),
    ("error code P0420", "error_handling"),
    ("DTC P0171 fuel system", "error_handling"),
    ("check engine light", "error_handling"),
    ("ABS warning", "error_handling"),
    ("ESC malfunction", "error_handling"),
    ("tire pressure alert", "error_handling"),
    ("battery low warning", "battery_warning"),
    ("battery management module fault", "battery_warning"),
    ("play music via bluetooth", "music_control"),
    ("next track", "music_control"),
    ("set volume to 60 percent", "music_control"),
    ("pause the music", "music_control"),
    ("stream audio USB", "music_control"),
    ("navigate to airport", "navigation"),
    ("route to nearest charging station", "navigation"),
    ("show traffic on route", "navigation"),
    ("what is the current speed", "query_sensor"),
    ("fuel level reading", "query_sensor"),
    ("show trip history", "query_historical"),
    ("set oil change alert", "set_alert"),
    ("calculate square root", "ood"),
    ("tell me a joke", "ood"),
]

# ---------------------------------------------------------------------------
# Pattern-based overrides (always take precedence over TF-IDF)
# ---------------------------------------------------------------------------
_PATTERN_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(dtc|p\d{4}|obd|check engine|fault code)\b", re.I), "error_handling"),
    (re.compile(r"\btpms\b|\btire pressure\b|\bflat tyre\b", re.I), "error_handling"),
    (re.compile(r"\babs\b|\besc\b|\bairbag\b|\bbrake warning\b", re.I), "error_handling"),
    (re.compile(r"\bbattery (low|warning|fault|management)\b", re.I), "battery_warning"),
    (re.compile(r"\b(temperature|hvac|ac|fan speed|heat|cool|defrost|seat heat)\b", re.I), "climate_control"),
    (re.compile(r"\b(play|pause|track|volume|music|audio|radio|bluetooth|usb)\b", re.I), "music_control"),
    (re.compile(r"\b(navigate|route|directions?|gps|destination|traffic|eta)\b", re.I), "navigation"),
    (re.compile(r"\b(speed|fuel level|rpm|coolant|sensor|psi)\b", re.I), "query_sensor"),
]


class CpuNLU:
    """TF-IDF + cosine-similarity intent classifier (CPU-only, sklearn).

    Falls back to pattern rules when sklearn is unavailable, and finally to
    a keyword-based heuristic as the last resort.
    """

    def __init__(self, data_paths: List[str] | None = None) -> None:
        self._vectorizer = None
        self._tfidf_matrix = None
        self._labels: List[str] = []
        self._fitted = False

        examples = self._load_examples(data_paths)
        self._fit(examples)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, text: str) -> Tuple[str, float]:
        """Return (intent_label, confidence) for *text*."""
        # 1. Pattern override
        for pattern, intent in _PATTERN_RULES:
            if pattern.search(text):
                return intent, 0.85

        # 2. TF-IDF cosine similarity
        if self._fitted and self._vectorizer is not None:
            try:
                import numpy as np  # type: ignore
                vec = self._vectorizer.transform([text])
                sims = (self._tfidf_matrix @ vec.T).toarray().flatten()
                best_idx = int(np.argmax(sims))
                score = float(sims[best_idx])
                if score > 0.05:
                    return self._labels[best_idx], min(score, 0.99)
            except Exception as e:  # pragma: no cover
                logger.debug("TF-IDF predict failed: %s", e)

        # 3. Keyword heuristic (no external deps)
        return self._keyword_fallback(text)

    def process(self, text: str) -> Dict:
        """Return NLU dict compatible with QuantizedNLUPipeline.process()."""
        label, score = self.predict(text)
        return {"label": label, "score": score, "raw_text": text}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_examples(self, data_paths: List[str] | None) -> List[Tuple[str, str]]:
        """Load (text, intent) pairs from JSONL files."""
        if data_paths is None:
            repo_root = Path(__file__).parent.parent
            data_paths = [
                str(repo_root / "data" / "automotive_domain_train.jsonl"),
                str(repo_root / "data" / "bosch_dataset_seed.jsonl"),
            ]

        examples: List[Tuple[str, str]] = list(_FALLBACK_EXAMPLES)
        for path in data_paths:
            p = Path(path)
            if not p.exists():
                continue
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        text = d.get("text", "")
                        intent = d.get("intent", "")
                        if text and intent:
                            examples.append((text, intent))
                    except json.JSONDecodeError:
                        pass
        logger.info("CpuNLU loaded %d training examples", len(examples))
        return examples

    def _fit(self, examples: List[Tuple[str, str]]) -> None:
        """Fit TF-IDF vectoriser on training corpus."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            import numpy as np  # type: ignore

            texts = [t for t, _ in examples]
            self._labels = [l for _, l in examples]

            self._vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
            )
            self._tfidf_matrix = self._vectorizer.fit_transform(texts)
            self._fitted = True
            logger.info("CpuNLU TF-IDF fitted (%d examples, %d features)",
                        len(texts), self._tfidf_matrix.shape[1])
        except ImportError:
            logger.warning("sklearn not available; CpuNLU will use pattern + keyword fallback only")

    def _keyword_fallback(self, text: str) -> Tuple[str, float]:
        text_lower = text.lower()
        kw_map = [
            (["temperature", "hvac", "fan", "heat", "cool", "ac ", "defrost"], "climate_control"),
            (["error", "dtc", "obd", "fault", "malfunction", "check engine"], "error_handling"),
            (["battery", "voltage"], "battery_warning"),
            (["play", "music", "volume", "track", "audio", "radio"], "music_control"),
            (["navigate", "route", "direction", "destination", "traffic"], "navigation"),
            (["speed", "fuel", "rpm", "sensor", "pressure"], "query_sensor"),
        ]
        for keywords, intent in kw_map:
            if any(kw in text_lower for kw in keywords):
                return intent, 0.6
        return "ood", 0.3
