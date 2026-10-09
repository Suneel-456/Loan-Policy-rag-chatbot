"""Ingestion: load policy files -> clean -> chunk -> embed -> save the vector index.

Run:  python -m src.ingest
"""
import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src import config
from src.embeddings import get_embeddings

log = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Remove repeated page footers and normalise blank lines."""
    patterns = [re.compile(p) for p in config.BOILERPLATE_PATTERNS]
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if any(p.match(stripped) for p in patterns):
            continue
        kept.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def load_documents(policy_dir: Path = None) -> list[Document]:
    """Read every PDF (one Document per page) and TXT file in the policy folder."""
    policy_dir = Path(policy_dir or config.POLICY_DIR)
    files = sorted(p for p in policy_dir.glob("*") if p.suffix.lower() in {".pdf", ".txt"})
    pdf_stems = {p.stem for p in files if p.suffix.lower() == ".pdf"}

    docs: list[Document] = []
    for path in files:
        if path.suffix.lower() == ".txt":
            if path.stem in pdf_stems:  # the PDF version wins, avoids duplicate chunks
                continue
            text = clean_text(path.read_text(encoding="utf-8"))
            if text:
                docs.append(Document(page_content=text, metadata={"source": path.name}))
        else:
            for page_no, page in enumerate(PdfReader(str(path)).pages, start=1):
                text = clean_text(page.extract_text() or "")
                if text:
                    docs.append(
                        Document(page_content=text, metadata={"source": path.name, "page": page_no})
                    )
    if not docs:
        raise FileNotFoundError(f"No readable .pdf or .txt policy files found in {policy_dir}")
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        # prefer splitting at section boundaries so a chunk keeps its section heading
        separators=["\nSection ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def build_index() -> tuple[int, int]:
    """Build the vector index and save it to disk. Returns (pages, chunks)."""
    docs = load_documents()
    chunks = split_documents(docs)
    store = InMemoryVectorStore.from_documents(chunks, get_embeddings())
    config.VECTORSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    store.dump(str(config.VECTORSTORE_PATH))
    log.info("Indexed %d pages into %d chunks -> %s", len(docs), len(chunks), config.VECTORSTORE_PATH)
    return len(docs), len(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pages, n_chunks = build_index()
    print(f"Done: {pages} pages -> {n_chunks} chunks saved to {config.VECTORSTORE_PATH}")