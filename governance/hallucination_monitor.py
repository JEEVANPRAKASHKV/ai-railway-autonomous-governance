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

# Words that carry no factual claim (English + Hinglish function words + assistant boilerplate).
_STOPWORDS = frozenset("""
the and for are was were you your yours not but with this that these those from have has had will would can could should may might
any all its into out over under than then them they their there here what when which while who whom how why also only more most
some such each per via been being does did doing done get gets got use used using one two within without about after before
please visit contact call details detail information info help helpline website irctc railmadad railway railways indian
yes sure note kindly thank thanks regards ticket tickets passenger passengers
hai hain ka ki ke ko se me mein par pe aur ya toh to bhi nahi nahin kya kyun kyu jab tab agar aap aapka aapki aapke aapko apna apni
yeh ye woh wo is us isko usko hota hoti hote hua hui hue kar karo karein karna karne kiya liye lie sakte sakta sakti milta milti milega
mil jata jati jate raha rahi rahe wala wali wale tak sath saath jaankari zyaada zyada sampark depend karta baat baar
""".split())


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _content_tokens(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower().replace(",", ""))
    return [_stem(t) for t in toks if (len(t) >= 3 or t.isdigit()) and t not in _STOPWORDS]


_NEGATION_WORDS = frozenset({"not", "no", "never", "cannot", "non", "without", "nahi", "nahin", "na", "nor",
                             "none", "neither", "cant", "wont", "isnt", "arent", "dont", "doesnt", "unavailable",
                             "ineligible", "prohibited"})


_CLAUSE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|[;:,()]\s*|\s+(?:but|however|although|though|except|unless|whereas|lekin|magar)\s+",
                           re.IGNORECASE)


def _clauses(text: str) -> List[str]:
    return [c.strip() for c in _CLAUSE_SPLIT.split(text) if c and len(c.strip()) > 3]


def _negated(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower().replace("'", ""))
    return sum(1 for w in words if w in _NEGATION_WORDS) % 2 == 1


def _numbers(text: str) -> List[str]:
    return re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))


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

    def reset(self) -> None:
        with self._lock:
            self._history.clear()
            self._module_stats.clear()

    def _support(self, text: str, context_docs: List[str]) -> Tuple[float, float, int]:
        """Return (content-token support, numeric-fact support, n_numbers) of `text` against the context union."""
        ctx = " ".join(context_docs)
        ctx_tokens = set(_content_tokens(ctx))
        toks = _content_tokens(text)
        support = (sum(1 for t in toks if t in ctx_tokens) / len(toks)) if toks else 1.0
        nums = _numbers(text)
        ctx_nums = set(_numbers(ctx))
        num_support = (sum(1 for n in nums if n in ctx_nums) / len(nums)) if nums else 1.0
        return support, num_support, len(nums)

    def _compute_faithfulness(self, response: str, context_docs: List[str]) -> float:
        """
        Faithfulness = how much of what the response *claims* is supported by the retrieved context:
        70% content-word support + 30% numeric-fact support (amounts, hours, days must appear in the source).
        An unsupported number caps the score — invented figures are the classic hallucination.
        """
        if not context_docs:
            return 0.0
        support, num_support, n_nums = self._support(response, context_docs)
        score = 0.7 * support + 0.3 * num_support
        if n_nums and num_support < 1.0:
            score = min(score, 0.5 * (1 + num_support) * 0.7)
        return float(round(min(score, 1.0), 4))

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
        Polarity check: for each answer sentence, find the policy sentence that talks about the same
        thing (high content-word overlap) and flag it when exactly one of them is negated
        ("refundable" vs "non-refundable", "a full refund" vs "no refund"). Word-overlap scores alone
        cannot see this — an answer with the opposite meaning but the same words used to score 1.0.
        """
        ctx_sents = [c.strip() for doc in context_docs for c in _clauses(doc)]
        found: List[str] = []
        for sent in _clauses(response):
            st = set(_content_tokens(sent)) - _NEGATION_WORDS
            if len(st) < 2:
                continue
            best, best_j = None, 0.0
            for cs in ctx_sents:
                ct = set(_content_tokens(cs)) - _NEGATION_WORDS
                if not ct:
                    continue
                j = len(st & ct) / len(st | ct)
                if j > best_j:
                    best, best_j = cs, j
            if best is not None and best_j >= 0.4 and _negated(sent) != _negated(best):
                found.append(f"'{sent.strip()[:90]}' contradicts policy: '{best.strip()[:90]}'")
        return found

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
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", response) if len(_content_tokens(s)) >= 4]
        if not sentences or not context_docs:
            return 1.0, []

        ungrounded = []
        for sent in sentences:
            support, num_support, n_nums = self._support(sent, context_docs)
            if support < 0.5 or (n_nums and num_support < 1.0):
                ungrounded.append(sent)

        grounded_count = len(sentences) - len(ungrounded)
        return round(grounded_count / len(sentences), 4), ungrounded

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
        composite = round(0.6 * faithfulness + 0.4 * groundedness_score, 4)

        contradictions = self._detect_contradictions(response, context_docs)
        harmful = self._detect_harmful_content(response)
        if contradictions:
            # A direct contradiction of policy is the most harmful kind of hallucination.
            composite = round(min(composite, 0.15), 4)

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
