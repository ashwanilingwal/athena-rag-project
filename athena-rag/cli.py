"""Manual poking without a Java client. Not part of the deliverable, but the
fastest way to sanity-check a change before writing a test for it.

  EMBED_PROVIDER=hash python cli.py index <tenant-uuid> ../corpus
  EMBED_PROVIDER=hash python cli.py search <tenant-uuid> "reciprocal rank fusion"
  EMBED_PROVIDER=hash python cli.py ask <tenant-uuid> "what is chunk overlap"
"""
import sys, uuid
from pathlib import Path
from athena_rag import Athena

def main() -> None:
    cmd, tenant, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    a = Athena()
    if cmd == "index":
        for p in sorted(Path(rest[0]).glob("*.md")):
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{tenant}/{p.name}"))
            r = a.index_document(tenant, doc_id, str(p))
            print(f"{p.name:20s} {r.chunks_indexed:3d} chunks  {r.took_ms:5d}ms")
    elif cmd == "search":
        strategy = rest[1] if len(rest) > 1 else "HYBRID_RRF"
        r = a.retrieve(tenant, rest[0], strategy=strategy)
        for i, p in enumerate(r.passages, 1):
            print(f"[{i}] {p.score:.4f} {p.metadata.get('heading','')}\n    {p.text[:110]}...\n")
        print(f"embed={r.embed_ms}ms vector={r.vector_ms}ms lexical={r.lexical_ms}ms rerank={r.rerank_ms}ms")
    elif cmd == "ask":
        for kind, payload in a.answer_stream(tenant, rest[0]):
            if kind == "sources":
                print("sources:", ", ".join(p.metadata.get("heading", "?") for p in payload.passages), "\n")
            elif kind == "token":
                print(payload, end="", flush=True)
            else:
                print(f"\n\n[grounded={payload['grounded']} generate={payload['generate_ms']}ms]")

if __name__ == "__main__":
    main()
