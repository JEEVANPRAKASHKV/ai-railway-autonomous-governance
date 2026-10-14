"""
governance/hallucination_monitor.py
─────────────────────────────────────
Pillar 4 – LLM Hallucination & Faithfulness Monitor.

Implements:
  • Faithfulness scoring via token-level overlap with retrieved context
  • Groundedness check — ensures LLM answer is attributable to source documents
  • Contradiction detection — flags responses that contradict policy documents
  • Confidence calibration — compares LLM's expressed certainty vs. retrieval confidence
  • Toxicity / harmful content pattern detection
  • Automatic quarantine of low-faithfulness responses
"""

from __future__ import annotations

import re
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from governance.telemetry_bus import telemetry_bus


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

FAITHFULNESS_THRESHOLD = 0.35       # Below this → likely hallucination
GROUNDEDNESS_THRESHOLD = 0.25       # Minimum cosine similarity to source docs
QUARANTINE_THRESHOLD = 0.20         # Automatic quarantine below this score
CONTRADICTION_KEYWORDS = [
    "however", "but actually", "contrary to", "incorrect", "wrong", "false",
    "not true", "misinformation", "that is incorrect",
]
HARMFUL_PATTERNS = [
    r"\b(hack|exploit|bypass|illegal)\b",
    r"\b(bomb|weapon|violence)\b",
    r"\b(scam|fraud|phishing)\b",
]
HISTORY_SIZE = 500


# ──────────────────────────────────────────────────────────────────────────────
# Hallucination Monitor
# ──────────────────────────────────────────────────────────────────────────────

class HallucinationMonitor:
    """
    Evaluates LLM-generated responses for faithfulness to retrieved context
    and detects hallucinations, groundedness violations, and harmful content.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._history: deque = deque(maxlen=HISTORY_SIZE)
        self._module_stats: Dict[str, dict] = {}
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")
        self._corpus_fitted = False

    # ── Scoring Primitives ────────────────────────────────────────────────────

    def _compute_faithfulness(self, response: str, context_docs: List[str]) -> float:
        """
        Faithfulness = max cosine similarity between response and any context doc.
        Uses TF-IDF vectors on unigrams+bigrams.
        """
        if not context_docs:
            return 0.0

        all_texts = context_docs + [response]
        try:
            tfidf_matrix = self._vectorizer.fit_transform(all_texts)
            context_vecs = tfidf_matrix[:-1]
            response_vec = tfidf_matrix[-1]
            sims = cosine_similarity(response_vec, context_vecs)[0]
            return float(round(np.max(sims), 4))
        except Exception:
            return 0.0

    def _compute_token_overlap(self, response: str, context: str) -> float:
        """Word-level Jaccard overlap as a secondary signal."""
        resp_tokens = set(response.lower().split())
        ctx_tokens = set(context.lower().split())
        if not ctx_tokens:
            return 0.0
        intersection = resp_tokens & ctx_tokens
        union = resp_tokens | ctx_tokens
        return round(len(intersection) / len(union), 4) if union else 0.0

    def _detect_contradictions(self, response: str, context_docs: List[str]) -> List[str]:
        """
        Heuristic contradiction detection:
        - Flags contradiction marker phrases in the response
        - Returns list of detected contradiction signals
        """
        response_lower = response.lower()
        detected = [kw for kw in CONTRADICTION_KEYWORDS if kw in response_lower]
        return detected

    def _detect_harmful_content(self, response: str) -> List[str]:
        """Regex-based harmful content / policy violation detection."""
        found = []
        for pattern in HARMFUL_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                found.append(pattern)
        return found

    def _compute_groundedness_score(
        self, response: str, context_docs: List[str]
    ) -> Tuple[float, List[str]]:
        """
        Groundedness: fraction of response sentences that can be attributed
        to at least one context document (similarity ≥ GROUNDEDNESS_THRESHOLD).
        Returns (score, ungrounded_sentences).
        """
        sentences = [s.strip() for s in re.split(r"[.!?]", response) if len(s.strip()) > 20]
        if not sentences or not context_docs:
            return 1.0, []

        ungrounded = []
        for sent in sentences:
            try:
                texts = context_docs + [sent]
                mat = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, stop_words="english").fit_transform(texts)
                sims = cosine_similarity(mat[-1], mat[:-1])[0]
                if np.max(sims) < GROUNDEDNESS_THRESHOLD:
                    ungrounded.append(sent)
            except Exception:
                pass

        grounded_count = len(sentences) - len(ungrounded)
        score = round(grounded_count / len(sentences), 4) if sentences else 1.0
        return score, ungrounded

    # ── Main Evaluation ───────────────────────────────────────────────────────

    def evaluate(
        self,
        module_id: str,
        query: str,
        response: str,
        context_docs: List[str],
        model_confidence: Optional[float] = None,
    ) -> dict:
        """
        Full hallucination evaluation for one LLM response.

        Args:
            module_id: Which LLM module generated this response.
            query: The user's original query.
            response: The LLM-generated response text.
            context_docs: Retrieved source documents used for grounding.
            model_confidence: Optional self-reported confidence from the LLM.

        Returns:
            dict with faithfulness score, groundedness, quarantine flag, recommendations.
        """
        faithfulness = self._compute_faithfulness(response, context_docs)
        groundedness_score, ungrounded_sentences = self._compute_groundedness_score(
            response, context_docs
        )

        # Token overlap (secondary)
        overlap = (
            max(self._compute_token_overlap(response, doc) for doc in context_docs)
            if context_docs else 0.0
        )

        # Composite score
        composite = round(
            0.5 * faithfulness + 0.3 * groundedness_score + 0.2 * overlap, 4
        )

        contradictions = self._detect_contradictions(response, context_docs)
        harmful = self._detect_harmful_content(response)

        quarantine = composite < QUARANTINE_THRESHOLD or bool(harmful)
        severity = (
            "CRITICAL" if quarantine
            else "WARNING" if composite < FAITHFULNESS_THRESHOLD
            else "INFO"
        )

        # Confidence calibration
        calibration_gap = (
            round(abs(model_confidence - composite), 4)
            if model_confidence is not None
            else None
        )

        result = {
            "module_id": module_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query[:200],
            "faithfulness_score": faithfulness,
            "groundedness_score": groundedness_score,
            "token_overlap": overlap,
            "composite_score": composite,
            "quarantine": quarantine,
            "severity": severity,
            "contradictions_detected": contradictions,
            "harmful_content_detected": harmful,
            "ungrounded_sentences": ungrounded_sentences[:3],  # top 3
            "model_confidence": model_confidence,
            "calibration_gap": calibration_gap,
            "recommendation": self._recommendation(composite, harmful, contradictions),
        }

        with self._lock:
            self._history.append(result)
            stats = self._module_stats.setdefault(module_id, {
                "total_evaluations": 0,
                "quarantined": 0,
                "avg_faithfulness": 0.0,
                "avg_composite": 0.0,
            })
            n = stats["total_evaluations"]
            stats["avg_faithfulness"] = round(
                (stats["avg_faithfulness"] * n + faithfulness) / (n + 1), 4
            )
            stats["avg_composite"] = round(
                (stats["avg_composite"] * n + composite) / (n + 1), 4
            )
            stats["total_evaluations"] += 1
            if quarantine:
                stats["quarantined"] += 1

        if severity in ("WARNING", "CRITICAL"):
            telemetry_bus.publish(
                module_id=module_id,
                event_type="HALLUCINATION_DETECTED" if quarantine else "LOW_FAITHFULNESS",
                payload={
                    "composite_score": composite,
                    "faithfulness": faithfulness,
                    "quarantine": quarantine,
                    "harmful": bool(harmful),
                },
                severity=severity,
            )

        return result

    @staticmethod
    def _recommendation(score: float, harmful: List, contradictions: List) -> str:
        recs = []
        if harmful:
            recs.append("BLOCK: Response contains potentially harmful content — do not serve.")
        if score < QUARANTINE_THRESHOLD:
            recs.append("QUARANTINE: Low faithfulness — return fallback policy response.")
        elif score < FAITHFULNESS_THRESHOLD:
            recs.append("REVIEW: Consider adding disclaimer or retrieving additional context.")
        if contradictions:
            recs.append("CONTRADICTION: Response may contradict source documents — verify manually.")
        return " | ".join(recs) if recs else "Response appears faithful and grounded."

    def get_module_stats(self, module_id: str) -> dict:
        with self._lock:
            return self._module_stats.get(module_id, {})

    def get_platform_stats(self) -> dict:
        with self._lock:
            return {
                "modules": dict(self._module_stats),
                "total_evaluations": sum(
                    s["total_evaluations"] for s in self._module_stats.values()
                ),
                "total_quarantined": sum(
                    s["quarantined"] for s in self._module_stats.values()
                ),
            }

    def get_recent_evaluations(self, n: int = 20) -> List[dict]:
        with self._lock:
            return list(self._history)[-n:]


# ── Singleton export ──────────────────────────────────────────────────────────
hallucination_monitor = HallucinationMonitor()
