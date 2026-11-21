#!/usr/bin/env python3
"""Deterministic, auditable baseline router for the ten CHAKRA experts.

This is a routing baseline, not a learned classifier. It does not load or alter
checkpoints and has no training side effects.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

DOMAINS = (
    "science", "math", "reasoning", "news", "qa",
    "general", "coding", "document", "history", "geography",
)

# Phrase weights are deliberately explicit so every decision is inspectable.
KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "science": (
        "chemistry", "chemical", "physics", "biology", "molecule", "reaction",
        "atom", "periodic table", "organic chemistry", "inorganic chemistry",
        "electrochemistry", "thermodynamics", "जीवविज्ञान", "रसायन", "रसायन विज्ञान",
        "अभिक्रिया", "अणु", "परमाणु", "भौतिकी", "आवर्त सारणी", "ऊष्मागतिकी",
    ),
    "math": (
        "mathematics", "math", "algebra", "calculus", "equation", "integral",
        "derivative", "geometry", "probability", "statistics", "गणित", "बीजगणित",
        "कलन", "समाकलन", "अवकलज", "समीकरण", "प्रायिकता", "सांख्यिकी",
    ),
    "reasoning": (
        "reasoning", "logic", "logical", "deduce", "deduction", "proof",
        "puzzle", "analogy", "critical thinking", "तर्क", "तार्किक", "निष्कर्ष",
        "पहेली", "सिद्ध कीजिए", "प्रमाणित", "तर्कशक्ति",
    ),
    "news": (
        "latest news", "breaking news", "current affairs", "today's news",
        "headline", "news update", "समसामयिक", "करंट अफेयर्स", "ताजा खबर",
        "ताज़ा खबर", "आज की खबर", "समाचार", "न्यूज़",
    ),
    "coding": (
        "python", "javascript", "typescript", "java code", "rust", "programming",
        "code", "debug", "compiler", "api", "function", "class", "sql",
        "कोड", "प्रोग्रामिंग", "डिबग", "कंपाइलर", "फंक्शन", "एल्गोरिदम",
    ),
    "document": (
        "summarize this document", "document", "pdf", "report", "resume",
        "cv", "draft an email", "write an email", "letter", "proposal",
        "अनुच्छेद", "दस्तावेज", "दस्तावेज़", "पीडीएफ", "रिपोर्ट", "पत्र",
        "आवेदन", "ईमेल", "सारांश",
    ),
    "history": (
        "history", "historical", "ancient", "medieval", "modern history",
        "civilization", "dynasty", "empire", "इतिहास", "प्राचीन", "मध्यकालीन",
        "आधुनिक इतिहास", "सभ्यता", "वंश", "साम्राज्य",
    ),
    "geography": (
        "geography", "geographical", "climate", "river", "mountain", "latitude",
        "longitude", "continent", "map", "भूगोल", "जलवायु", "नदी", "पर्वत",
        "अक्षांश", "देशांतर", "महाद्वीप", "मानचित्र",
    ),
    "qa": (
        "multiple choice", "mcq", "question answer", "question bank",
        "quiz me", "प्रश्नोत्तरी", "बहुविकल्पीय", "वस्तुनिष्ठ प्रश्न",
        "प्रश्न बैंक", "क्विज",
    ),
    "general": (
        "general knowledge", " सामान्य ज्ञान", "सामान्य ज्ञान", "परिचय",
    ),
}

@dataclass(frozen=True)
class Route:
    expert: str
    confidence: float
    scores: Dict[str, int]
    matched: Dict[str, List[str]]
    reason: str

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

def route_query(query: str, min_score: int = 2) -> Route:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if min_score < 1:
        raise ValueError("min_score must be >= 1")

    text = " " + normalize(query) + " "
    scores = {domain: 0 for domain in DOMAINS}
    matched: Dict[str, List[str]] = {domain: [] for domain in DOMAINS}

    for domain, phrases in KEYWORDS.items():
        for phrase in phrases:
            needle = " " + normalize(phrase) + " "
            if needle in text:
                # Multiword/specific phrases receive a modestly higher weight.
                weight = 2 if (" " in phrase.strip() or domain == "qa") else 1
                scores[domain] += weight
                matched[domain].append(phrase.strip())

    ranked = sorted(DOMAINS, key=lambda d: (-scores[d], DOMAINS.index(d)))
    best = ranked[0]
    total = sum(scores.values())
    if scores[best] < min_score:
        return Route("general", 0.0, scores, matched,
                     "No domain reached the minimum evidence threshold; safe fallback to general.")
    confidence = scores[best] / total if total else 0.0
    reason = "Matched: " + ", ".join(matched[best])
    if len(ranked) > 1 and scores[ranked[1]] == scores[best]:
        reason += "; tie resolved by fixed domain order (review this ambiguous query)."
    return Route(best, round(confidence, 4), scores, matched, reason)

def main() -> None:
    parser = argparse.ArgumentParser(description="Route a query to one of ten CHAKRA experts.")
    parser.add_argument("query", nargs="+", help="Hindi or English query")
    parser.add_argument("--min-score", type=int, default=2)
    args = parser.parse_args()
    result = route_query(" ".join(args.query), args.min_score)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
