"""Text tools: summarize, extract entities, translate, diff texts."""

import re
from typing import Any

from stem_agent.tools.catalog import CatalogTool


def _summarize(input_data: dict[str, Any]) -> dict[str, Any]:
    """Extractive text summarization: pick top sentences by keyword density."""
    text = input_data.get("text", "")
    max_sentences = input_data.get("max_sentences", 3)

    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= max_sentences:
        return {"summary": text, "original_sentences": len(sentences)}

    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    if not words:
        return {"summary": " ".join(sentences[:max_sentences])}

    word_freq = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1

    scored = []
    for i, s in enumerate(sentences):
        sw = re.findall(r'\b[a-zA-Z]{4,}\b', s.lower())
        if sw:
            score = sum(word_freq.get(w, 0) for w in sw) / len(sw)
            scored.append((score, i, s))

    scored.sort(reverse=True)
    top = sorted(scored[:max_sentences], key=lambda x: x[1])  # restore order
    summary = " ".join(s[2] for s in top)
    return {"summary": summary, "original_sentences": len(sentences)}


def _extract_entities(input_data: dict[str, Any]) -> dict[str, Any]:
    """Extract named entities: emails, URLs, dates, money, percentages."""
    text = input_data.get("text", "")

    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    urls = re.findall(r'https?://[^\s<>"]+', text)
    # Strip trailing punctuation
    urls = [u.rstrip('.,;:!?)]') for u in urls]
    dates = re.findall(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}', text)
    money = re.findall(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?', text)
    percentages = re.findall(r'\d+(?:\.\d+)?%', text)

    return {
        "emails": list(set(emails)),
        "urls": list(set(urls)),
        "dates": list(set(dates)),
        "money": list(set(money)),
        "percentages": list(set(percentages)),
    }


def _translate(input_data: dict[str, Any]) -> dict[str, Any]:
    """Placeholder for translation. Returns input as-is (requires API key for real use)."""
    text = input_data.get("text", "")
    target_lang = input_data.get("target_language", "en")
    return {"translated": text, "target_language": target_lang, "note": "Translation requires LLM integration"}


def _diff_texts(input_data: dict[str, Any]) -> dict[str, Any]:
    """Compute word-level diff between two texts. Returns added/removed words."""
    import difflib
    text_a = input_data.get("text_a", "").split()
    text_b = input_data.get("text_b", "").split()

    diff = list(difflib.unified_diff(text_a, text_b, lineterm=""))
    added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))

    return {
        "words_added": added,
        "words_removed": removed,
        "similarity": difflib.SequenceMatcher(None, text_a, text_b).ratio(),
    }


def register(catalog):
    catalog.register(CatalogTool(
        name="summarize",
        description="Extractive text summary: pick top N sentences by keyword density.",
        category="text",
        parameter_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to summarize"},
                "max_sentences": {"type": "integer", "description": "Max output sentences", "default": 3},
            },
            "required": ["text"],
        },
        examples=[
            {"input": 'text="Long article text..." max_sentences=2', "output": "{summary: '...'}"},
        ],
        cost_estimate=0.0,
        fn=_summarize,
    ))
    catalog.register(CatalogTool(
        name="extract_entities",
        description="Extract emails, URLs, dates, money, percentages from text.",
        category="text",
        parameter_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to scan"},
            },
            "required": ["text"],
        },
        examples=[
            {"input": 'text="Contact alice@co.com. Budget $5,000 by 2026-06-01."',
             "output": "{emails: ['alice@co.com'], money: ['$5,000'], dates: ['2026-06-01']}"},
        ],
        cost_estimate=0.0,
        fn=_extract_entities,
    ))
    catalog.register(CatalogTool(
        name="diff_texts",
        description="Word-level diff between two texts. Returns similarity ratio + word changes.",
        category="text",
        parameter_schema={
            "type": "object",
            "properties": {
                "text_a": {"type": "string", "description": "Original text"},
                "text_b": {"type": "string", "description": "Modified text"},
            },
            "required": ["text_a", "text_b"],
        },
        examples=[
            {"input": 'text_a="hello world" text_b="hello new world"',
             "output": "{words_added:1, similarity:0.8}"},
        ],
        cost_estimate=0.0,
        fn=_diff_texts,
    ))
