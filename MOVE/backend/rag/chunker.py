from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Any


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


def chunk_text(text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Split text into overlapping chunks and attach metadata to each."""
    docs = _splitter.create_documents([text])
    return [
        {
            "content": doc.page_content,
            "metadata": metadata or {},
            "chunk_index": i,
        }
        for i, doc in enumerate(docs)
    ]


def chunk_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chunk a list of scraped pages, propagating per-page metadata."""
    all_chunks = []
    for page in pages:
        content = page.get("content", "")
        if not content:
            continue
        meta = {
            "url": page.get("url", ""),
            "source": page.get("source", ""),
            **page.get("metadata", {}),
        }
        all_chunks.extend(chunk_text(content, meta))
    return all_chunks
