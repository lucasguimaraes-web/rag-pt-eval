"""RAG mínimo em português: ingestão -> chunks -> embeddings -> busca -> resposta com citação.

Sem banco vetorial de propósito: numpy basta para centenas de documentos e deixa o código legível.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMBED_MODEL = os.environ.get("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")


@dataclass
class Chunk:
    source: str
    text: str


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return [c for c in chunks if len(c) > 50]


class Index:
    def __init__(self, docs_dir: str = "data/docs", chunk_size: int = 800, overlap: int = 150):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.chunks: list[Chunk] = []
        for path in sorted(Path(docs_dir).glob("*")):
            if path.suffix.lower() in {".pdf", ".txt", ".md"}:
                for piece in chunk_text(read_file(path), chunk_size, overlap):
                    self.chunks.append(Chunk(path.name, piece))
        if not self.chunks:
            raise RuntimeError(f"Nenhum documento .pdf/.txt/.md encontrado em {docs_dir}")
        self.emb = self.model.encode(
            [c.text for c in self.chunks], normalize_embeddings=True, show_progress_bar=False
        )

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.emb @ q
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


def answer(index: Index, question: str, k: int = 5) -> dict:
    """Gera resposta usando só o contexto recuperado. Exige ANTHROPIC_API_KEY."""
    import anthropic

    hits = index.search(question, k)
    context = "\n\n".join(f"[{i + 1}] ({c.source}) {c.text}" for i, (c, _) in enumerate(hits))
    prompt = (
        "Responda em português, usando APENAS o contexto abaixo. "
        "Cite as fontes como [1], [2]. Se o contexto não tiver a resposta, diga que não encontrou.\n\n"
        f"Contexto:\n{context}\n\nPergunta: {question}"
    )
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=os.environ.get("LLM_MODEL", "claude-sonnet-5-5"),
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return {"answer": msg.content[0].text, "sources": [c.source for c, _ in hits]}


if __name__ == "__main__":
    import sys

    idx = Index()
    q = " ".join(sys.argv[1:]) or "Qual é o assunto principal dos documentos?"
    out = answer(idx, q)
    print(out["answer"], "\n\nFontes:", sorted(set(out["sources"])))
