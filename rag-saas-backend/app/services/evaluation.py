from typing import Dict, Any, List

class EvaluationService:
    """
    Reserved for runtime/online evaluation or quality logging within the backend API.
    Used to track, monitor, or log metrics from user inference runs.
    """
    def log_inference_run(self, query: str, answer: str, sources: List[str]) -> Dict[str, Any]:
        """
        Logs details of a single RAG (Retrieval-Augmented Generation) inference run.
        
        Args:
            query (str): The user's original input question.
            answer (str): The generated response provided by the LLM.
            sources (List[str]): A list of source identifiers or chunk references used to generate the answer.
            
        Returns:
            Dict[str, Any]: A dictionary containing structured log metadata for observability or monitoring.
        """
        return {
            # Capture the original user query for tracking evaluation datasets
            "query": query,
            
            # Capture the final generated answer to audit quality later
            "answer": answer,
            
            # Record how many source chunks were retrieved to evaluate retrieval breadth
            "retrieved_chunk_count": len(sources),
            
            # Provide a status flag indicating the inference run completed successfully
            "status": "completed"
        }

# Instantiate a global singleton service object to be imported across the application API routes
evaluation_service = EvaluationService()