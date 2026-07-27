"""
Idempotent ingestion CLI.

Usage:
    python ingest.py --input data/
    python ingest.py --input data/policy.pdf

Re-running on the same files is safe and does not create duplicate vectors: see
rag/chunking.py (deterministic chunk_id) and rag/store.py (upsert_document deletes
any existing chunks for a doc_id before inserting the current set).
"""
from __future__ import annotations
import argparse
import logging
from collections import defaultdict
from pathlib import Path

from config import get_settings
from rag.chunking import chunk_directory, chunk_document
from rag.store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="File or directory to ingest")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    chunk_size = args.chunk_size or settings.chunk_size_tokens
    overlap = args.overlap or settings.chunk_overlap_tokens

    path = Path(args.input)
    if path.is_dir():
        chunks = chunk_directory(path, chunk_size, overlap)
    else:
        chunks = chunk_document(path, chunk_size, overlap)

    if not chunks:
        log.warning("No chunks produced from %s (unsupported files or empty dir?)", path)
        return

    by_doc: dict[str, list] = defaultdict(list)
    for c in chunks:
        by_doc[c.doc_id].append(c)

    store = VectorStore(settings.chroma_persist_dir, settings.chroma_collection)

    total_inserted = 0
    for doc_id, doc_chunks in by_doc.items():
        n = store.upsert_document(doc_id, doc_chunks)
        total_inserted += n
        log.info("Ingested %-40s doc_id=%s chunks=%d", doc_chunks[0].source_path, doc_id, n)

    log.info(
        "Done. %d documents, %d chunks upserted (chunk_size=%d, overlap=%d). Collection size: %d",
        len(by_doc), total_inserted, chunk_size, overlap, store.count(),
    )


if __name__ == "__main__":
    main()
