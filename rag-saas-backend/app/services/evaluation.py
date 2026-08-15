from typing import Dict, Any, List

class EvaluationService:
    """
    Reserved for runtime/online evaluation or quality logging within the backend API.
    """
    def log_inference_run(self, query: str, answer: str, sources: List[str]) -> Dict[str, Any]:
        return {
            "query": query,
            "answer": answer,
            "retrieved_chunk_count": len(sources),
            "status": "completed"
        }

evaluation_service = EvaluationService()