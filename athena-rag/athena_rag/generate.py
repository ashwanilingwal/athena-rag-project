"""Prompt assembly and token streaming.

The grounding contract is enforced in two places: the system prompt tells the model
to refuse, and `looks_grounded` checks whether it did. Never trust the prompt alone
— the `grounded` flag on AnswerComplete is what Java logs and what your eval
harness measures faithfulness against.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Iterator

from .store import Hit

SYSTEM = (
    "Answer using ONLY the numbered context passages. Cite the passages you used "
    "inline like [1] or [2][3]. If the passages do not contain the answer, say "
    "exactly: I don't know based on the indexed documents. Be concise."
)

REFUSAL = "i don't know based on the indexed documents"


def build_prompt(query: str, passages: list[Hit], history: list[tuple[str, str]] | None = None):
    context = "\n\n".join(f"[{i + 1}] {p.text}" for i, p in enumerate(passages))
    messages = [{"role": "system", "content": SYSTEM}]
    for role, content in history or []:
        messages.append({"role": role, "content": content})
    messages.append(
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    )
    return messages


def looks_grounded(answer: str) -> bool:
    return REFUSAL not in answer.lower().strip()


class OllamaGenerator:
    def __init__(self, model: str, host: str) -> None:
        self.model = model
        self.host = host.rstrip("/")

    def stream(self, messages: list[dict]) -> Iterator[str]:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": 0},
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                if not line.strip():
                    continue
                frame = json.loads(line)
                token = frame.get("message", {}).get("content", "")
                if token:
                    yield token
                if frame.get("done"):
                    return


class ExtractiveGenerator:
    """Offline generator: emits the top passage's opening sentences with a citation.

    Lets the streaming path, the frame ordering, the grounding flag and the Java SSE
    mapping all be tested without an LLM. It is extractive, not generative — never
    put its output in a demo and call it an answer.
    """

    model = "extractive-offline"

    def stream(self, messages: list[dict]) -> Iterator[str]:
        user = messages[-1]["content"]
        context = user.split("Context:\n", 1)[-1].split("\n\nQuestion:")[0]
        first = context.strip().split("\n\n")[0] if context.strip() else ""
        if not first:
            yield "I don't know based on the indexed documents."
            return
        body = first.split("] ", 1)[-1]
        for word in (" ".join(body.split()[:60]) + " [1]").split(" "):
            yield word + " "
