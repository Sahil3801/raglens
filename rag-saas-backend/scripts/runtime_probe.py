"""Opt-in runtime smoke probe; no RAGAS, baseline comparison, or quality metrics.

Use a disposable local QDRANT_PATH and a collection starting with raglens_runtime_.
PDFs must be synthetic fixtures. The probe deletes only the documents it inserts.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import UploadFile
from app.core.config import settings
from app.repositories.vector_store import VectorStoreRepository
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.services.reranking import reranker_service
from app.services.generation import GenerationService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    if not settings.QDRANT_PATH or not settings.QDRANT_COLLECTION.startswith("raglens_runtime_"):
        parser.error("Use embedded Qdrant and a disposable raglens_runtime_ collection.")
    if len({p.name for p in args.pdf}) != len(args.pdf):
        parser.error("Fixture filenames must be distinct.")

    report = {
        "kind": "runtime_smoke_verification_not_evaluation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "packages": {p: version(p) for p in ("langchain-qdrant", "qdrant-client", "sentence-transformers", "langchain-groq", "pypdf")},
        "config": {k: getattr(settings, k) for k in ("GROQ_MODEL", "QDRANT_COLLECTION", "CHUNK_SIZE", "CHUNK_OVERLAP", "RETRIEVER_K", "RETRIEVER_FETCH_K", "RERANKER_TOP_N")},
        "qdrant_mode": "embedded_local",
        "documents": [],
        "query": args.query,
    }
    repo = VectorStoreRepository()
    if repo.client.count(settings.QDRANT_COLLECTION, exact=True).count:
        repo.client.close()
        parser.error("The disposable collection must be empty; refusing to alter existing data.")
    inserted = []
    try:
        vector = repo.embeddings.embed_query("RagLens actual embedding dimension check")
        report["actual_embedding_dimension"] = len(vector)
        assert len(vector) == settings.EMBEDDING_DIMENSION == 384
        ingestion = IngestionService(repo)
        for path in args.pdf:
            inserted.append(path.name)
            with path.open("rb") as stream:
                ingestion.process_pdf_upload(UploadFile(filename=path.name, file=stream))
            report["documents"].append({"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        points = []
        offset = None
        while True:
            batch, offset = repo.client.scroll(settings.QDRANT_COLLECTION, limit=100, offset=offset, with_vectors=True)
            points.extend(batch)
            if offset is None:
                break
        report["stored_points"] = len(points)
        report["stored_vector_dimensions"] = sorted({len(p.vector) for p in points})
        report["chunks_by_file"] = dict(Counter(p.payload["metadata"]["source_file"] for p in points))
        lengths = [len(p.payload["page_content"]) for p in points]
        report["chunk_character_lengths"] = {"min": min(lengths), "max": max(lengths)}
        report["page_metadata_preserved"] = all("page" in p.payload["metadata"] for p in points)
        report["temporary_upload_files_removed"] = all(not Path(f"temp_{p.name}").exists() for p in args.pdf)

        # Observe the real local Qdrant candidate search; do not replace its results.
        # This private observation point is tied to the package version recorded above.
        local_collection = repo.client._client.collections[settings.QDRANT_COLLECTION]
        original_search = local_collection.search
        def observed_search(*positional, **kwargs):
            candidates = original_search(*positional, **kwargs)
            report["candidate_search"] = {"requested": kwargs.get("limit"), "actual_count": len(candidates)}
            return candidates
        local_collection.search = observed_search
        original_query = repo.client.query_points
        def observed_query(*positional, **kwargs):
            response = original_query(*positional, **kwargs)
            report["mmr"] = {"requested_k": kwargs["limit"], "candidate_limit": kwargs["query"].mmr.candidates_limit,
                             "diversity": kwargs["query"].mmr.diversity, "returned": len(response.points)}
            return response
        repo.client.query_points = observed_query
        docs = RetrievalService(repo).retrieve(args.query)
        original_predict = reranker_service.encoder.predict
        def observed_predict(pairs, **kwargs):
            scores = original_predict(pairs, **kwargs)
            report["crossencoder"] = {"model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "scored_pairs": len(pairs),
                                      "scores": [float(s) for s in scores]}
            return scores
        reranker_service.encoder.predict = observed_predict
        selected = reranker_service.rerank(args.query, docs)
        report["llm_context_chunks"] = len(selected)
        report["selected_sources"] = [{"filename": d.metadata["source_file"], "page": d.metadata.get("page")} for d in selected]
        report["answer"] = GenerationService().generate_answer(args.query, selected)
        report["status"] = "completed"
    finally:
        for filename in inserted:
            repo.delete_by_source_file(filename)
        report["remaining_points_after_cleanup"] = repo.client.count(settings.QDRANT_COLLECTION, exact=True).count
        repo.client.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
