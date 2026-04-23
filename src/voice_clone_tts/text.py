from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    text = text.replace("*", "")
    text = text.replace("_", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" ?([,.;:!?…])", r"\1", text)
    return text.strip()


def split_paragraph(text: str, limit: int = 240) -> list[str]:
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        probe = f"{current} {sentence}".strip()
        if len(probe) <= limit or not current:
            current = probe
            continue

        chunks.append(current)
        current = sentence

    if current:
        chunks.append(current)
    return chunks


def split_text(
    text: str,
    *,
    chunk_limit: int = 240,
    paragraph_gap_ms: int = 850,
    sentence_gap_ms: int = 360,
) -> list[tuple[str, int]]:
    normalized = normalize_text(text)
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", normalized) if part.strip()]
    chunks: list[tuple[str, int]] = []

    for paragraph in paragraphs:
        paragraph_chunks = split_paragraph(paragraph, limit=chunk_limit)
        for idx, chunk in enumerate(paragraph_chunks):
            gap = paragraph_gap_ms if idx == len(paragraph_chunks) - 1 else sentence_gap_ms
            chunks.append((chunk, gap))

    if chunks:
        last_text, _ = chunks[-1]
        chunks[-1] = (last_text, 0)

    return chunks
