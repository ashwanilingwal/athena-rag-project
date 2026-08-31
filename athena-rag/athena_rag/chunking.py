"""Load a document off disk and split it into chunks that survive retrieval.

Two ideas worth defending in an interview:

1. Split on structure first, size second. A chunk that begins mid-sentence under a
   heading it never names is a chunk the embedder cannot represent well.
2. Carry the heading path into the chunk text. "Retry policy" is meaningless on its
   own; "MRTR > Error handling > Retry policy" embeds into a useful region of space.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ~4 characters per token is close enough for chunk sizing and avoids dragging a
# tokenizer dependency into the hot path. Be honest that it is an approximation.
CHARS_PER_TOKEN = 4

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    ordinal: int
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


def load(uri: str, content_type: str = "text/markdown") -> str:
    path = Path(uri)
    if not path.exists():
        raise FileNotFoundError(uri)
    text = path.read_text(encoding="utf-8", errors="replace")
    if content_type in ("text/markdown", "text/x-markdown"):
        text = _clean_markdown(text)
    return text


def _clean_markdown(text: str) -> str:
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)      # frontmatter
    text = re.sub(r"```.*?```", lambda m: m.group(0), text, flags=re.S)  # keep code
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)               # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)           # link -> label
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sections(text: str) -> list[tuple[list[str], str]]:
    """Split into (heading_path, body) pairs, tracking heading nesting."""
    out: list[tuple[list[str], str]] = []
    stack: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            out.append((list(stack), body))
        buf.clear()

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            stack = stack[: level - 1]
            stack.append(m.group(2).strip())
        else:
            buf.append(line)
    flush()
    return out or [([], text)]


def _split_sized(body: str, size: int, overlap: int) -> list[str]:
    """Pack paragraphs up to `size` chars, then back off by `overlap` chars."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        if len(p) > size:  # a single oversized paragraph: hard-split on sentences
            for sent in re.split(r"(?<=[.!?])\s+", p):
                if len(cur) + len(sent) + 1 > size and cur:
                    chunks.append(cur.strip())
                    cur = cur[-overlap:] if overlap else ""
                cur += sent + " "
            continue
        if len(cur) + len(p) + 2 > size and cur:
            chunks.append(cur.strip())
            cur = cur[-overlap:] if overlap else ""
        cur += p + "\n\n"
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def chunk(
    text: str,
    *,
    chunk_tokens: int = 512,
    chunk_overlap: int = 64,
    source: str = "",
) -> list[Chunk]:
    size = chunk_tokens * CHARS_PER_TOKEN
    overlap = chunk_overlap * CHARS_PER_TOKEN
    out: list[Chunk] = []

    for heading_path, body in _sections(text):
        # "Title > Title" reads as noise and wastes embedding budget.
        deduped = [h for i, h in enumerate(heading_path) if i == 0 or h != heading_path[i - 1]]
        header = " > ".join(deduped)
        for piece in _split_sized(body, size, overlap):
            body_text = f"{header}\n\n{piece}" if header else piece
            out.append(
                Chunk(
                    ordinal=len(out),
                    text=body_text,
                    metadata={"heading": header, "source": source},
                )
            )
    return out
