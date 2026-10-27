"""
modules/m6_support_assistant/rag_indexer.py
===========================================
Retriever for the Support Assistant over the verified knowledge base (data/knowledge/support_kb.json).

Each knowledge entry has a title, several ways passengers ask about it, a checked answer and its sources.
A query is matched against every phrasing with two TF-IDF views — whole words (1–2-grams) and character
n-grams (robust to spelling and word-form variants) — after expanding common synonyms. An entry's score is
its best-matching phrasing. Runs fully locally; no external API.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.config import settings

logger = logging.getLogger(__name__)

KB_PATH = os.path.join(settings.project_root, "data", "knowledge", "support_kb.json")

# Passenger wording → the vocabulary used in the knowledge base (appended, never replaced)
SYNONYMS = {
    r"\bmoney back\b|\breimburse\w*|\brefunded\b|\bget back my money\b|\bmoney returned\b": "refund",
    r"\bid\b|\bidentity\b|\bid card\b|\bdocuments?\b": "id proof identity",
    r"\bkids?\b|\binfants?\b|\bbab(y|ies)\b|\btoddler\b|\b\d{1,2}\s*(year|yr)s?[\s-]*old\b": "child children ticket age",
    r"\bwom[ae]n\b|\bfemale\b|\bgirls?\b|\blady\b": "ladies",
    r"\blate\b|\bdelay(ed)?\b|\brunning behind\b": "late delayed",
    r"\belderly\b|\bold age\b|\bold parents?\b|\bgrand(mother|father|parents?)\b|\baged\b": "senior citizen",
    r"\b3\s?ac\b|\bac 3\b|\bthird ac\b|\b3 tier\b": "3a",
    r"\b2\s?ac\b|\bac 2\b|\bsecond ac\b|\b2 tier\b": "2a",
    r"\b1\s?ac\b|\bfirst ac\b": "1a",
    r"\bwaiting\b|\bwait list\b|\bwl\b": "waitlist waitlisted",
    r"\bbaggage\b|\bbags?\b|\bsuitcase\b": "luggage",
    r"\bmeals?\b|\bfood\b|\beat\b|\bdinner\b|\blunch\b|\bbreakfast\b": "food meals catering",
    r"\bcomplain\w*\b|\bcustomer care\b|\bhelp ?line\b|\bcontact\b": "helpline complaint",
    r"\bcancel+ed\b|\bcancel+ation\b|\bcancel+ing\b": "cancel",
    r"\b(sister|brother|mother|father|son|daughter|wife|husband|spouse|relative)\b": "family member",
    r"\bon my ticket\b|\bsomeone else\b|\banother person\b": "transfer",
    r"\bahead\b|\bbeforehand\b|\bprior\b|\bhow early\b": "advance",
    r"\breserv(e|ing)\b": "book",
    r"\bofficial\b|\bgenuine\b|\breal (irctc|website|site|ticket)\b": "official real irctc",
}
# generic question words must not decide which topic matches
_STOP = ["the", "a", "an", "is", "are", "am", "i", "my", "me", "to", "of", "for", "in", "on", "and", "or", "it", "do", "does",
         "did", "be", "this", "that", "with", "at", "from", "will", "please", "we", "our", "what", "how", "can", "could",
         "which", "when", "where", "why", "there", "any", "get", "need", "about", "difference", "between", "explain", "tell",
         "know", "want", "should", "would", "much", "many", "if", "so", "just", "still", "has", "have", "been", "was",
         "mean", "means", "meaning"]


def normalise(text: str) -> str:
    t = re.sub(r"[^\w\s₹+]", " ", str(text).lower())
    extra = [rep for pat, rep in SYNONYMS.items() if re.search(pat, t)]
    return re.sub(r"\s+", " ", (t + " " + " ".join(extra)).strip())


class LocalRAGIndexer:
    """Knowledge-base retriever: entry-level scores from the best-matching phrasing."""

    def __init__(self, path: str = KB_PATH) -> None:
        self.path = path
        self.entries: List[Dict[str, Any]] = []
        self.version = ""
        self._word: Optional[TfidfVectorizer] = None
        self._char: Optional[TfidfVectorizer] = None
        self._build_index()

    @property
    def chunks(self) -> List[Dict[str, Any]]:      # backwards-compatible name
        return self.entries

    def _build_index(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                kb = json.load(f)
            self.entries = kb.get("entries", [])
            self.version = kb.get("version", "")
        except Exception as e:
            logger.error(f"[RAG] Could not load knowledge base {self.path}: {e}")
            self.entries = []
            return
        keys: List[str] = []
        key_entry: List[int] = []
        key_weight: List[float] = []
        for i, e in enumerate(self.entries):
            for q in [e.get("title", "")] + list(e.get("questions", [])):
                keys.append(normalise(q)); key_entry.append(i); key_weight.append(1.0)
            # the answer helps with rare content words, but counts less than a matching phrasing
            keys.append(normalise(e.get("answer", ""))); key_entry.append(i); key_weight.append(0.55)
        self._word = TfidfVectorizer(ngram_range=(1, 2), stop_words=_STOP, sublinear_tf=True)
        self._char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self._Mw = self._word.fit_transform(keys)
        self._Mc = self._char.fit_transform(keys)
        self._key_entry = np.array(key_entry)
        self._key_weight = np.array(key_weight)
        logger.info(f"[RAG] Knowledge base v{self.version}: {len(self.entries)} entries, {len(keys)} phrasings indexed.")

    def rebuild(self) -> None:
        self._build_index()

    def score_entries(self, query: str) -> np.ndarray:
        """Best phrasing score per entry, in [0, 1]."""
        if not self.entries or self._word is None:
            return np.zeros(0)
        q = normalise(query)
        sw = cosine_similarity(self._word.transform([q]), self._Mw)[0]
        sc = cosine_similarity(self._char.transform([q]), self._Mc)[0]
        key_scores = (0.5 * sw + 0.5 * sc) * self._key_weight
        scores = np.zeros(len(self.entries))
        np.maximum.at(scores, self._key_entry, key_scores)
        return scores

    def retrieve(self, query: str, top_k: int = 3, category: Optional[str] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Top-k knowledge entries as (chunk, score); a chunk carries id, title, answer, sources and category."""
        scores = self.score_entries(query)
        out: List[Tuple[Dict[str, Any], float]] = []
        for idx in np.argsort(scores)[::-1]:
            e = self.entries[int(idx)]
            if category and e.get("category") != category:
                continue
            out.append(({"chunk_id": e["id"], "source_file": e.get("title", e["id"]), "text": e.get("answer", ""),
                         "title": e.get("title", ""), "sources": e.get("sources", []), "category": e.get("category", "")},
                        round(float(scores[idx]), 4)))
            if len(out) >= top_k:
                break
        return out

    def get(self, entry_id: str) -> Optional[Dict[str, Any]]:
        return next((e for e in self.entries if e.get("id") == entry_id), None)


rag_indexer = LocalRAGIndexer()
