import sys
import types
import langchain_community
import asyncio
import json
import os
import requests
import pandas as pd
from dotenv import load_dotenv

# --------------------------------------------------
# Vertex AI compatibility mock for older Ragas builds
# --------------------------------------------------
chat_models_mock = types.ModuleType("chat_models")
sys.modules["langchain_community.chat_models"] = chat_models_mock
langchain_community.chat_models = chat_models_mock

vertexai_mock = types.ModuleType("vertexai")
vertexai_mock.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = vertexai_mock
chat_models_mock.vertexai = vertexai_mock
# --------------------------------------------------

from ragas.metrics.collections import Faithfulness, AnswerRelevancy
from ragas.embeddings import HuggingFaceEmbeddings
from openai import AsyncOpenAI
from ragas.llms import llm_factory

load_dotenv()

async def run_benchmark_suite(benchmarks_path: str = "evaluation/benchmarks/resume_benchmark.json"):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ Error: GROQ_API_KEY not found in .env")
        return

    # Initialize LLM and Embeddings
    groq_client = AsyncOpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
    ragas_llm = llm_factory("llama-3.1-8b-instant", client=groq_client)
    ragas_embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

    # Initialize Ragas Metrics
    faithfulness_metric = Faithfulness(llm=ragas_llm)
    answer_relevancy_metric = AnswerRelevancy(
        llm=ragas_llm, 
        embeddings=ragas_embeddings, 
        strictness=1
    )

    if not os.path.exists(benchmarks_path):
        print(f"❌ Error: Benchmark file not found at {benchmarks_path}")
        return

    with open(benchmarks_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    results = []
    print(f"🚀 Running Evaluation on {len(benchmarks)} benchmark items...\n")

    for idx, item in enumerate(benchmarks):
        # Safely extract keys, handling both "query" and "question" formats
        q = item.get("question") or item.get("query")
        category = item.get("category", "unassigned")
        item_id = item.get("id", str(idx + 1))

        print(f"Testing [{item_id}] ({category}): '{q}'")

        # Call your FastAPI backend
        try:
            res = requests.post("http://127.0.0.1:8000/chat", json={"query": q})
            res.raise_for_status()
            data = res.json()
            answer = data["answer"]
            contexts = data["sources"]
        except Exception as e:
            print(f"  ❌ API call failed: {e}")
            continue

        # Evaluate the response using Ragas
        f_score = await faithfulness_metric.ascore(
            user_input=q,
            response=answer,
            retrieved_contexts=contexts
        )
        
        ar_score = await answer_relevancy_metric.ascore(
            user_input=q,
            response=answer
        )

        # Extract numerical values safely
        f_val = f_score.value if hasattr(f_score, 'value') else f_score
        ar_val = ar_score.value if hasattr(ar_score, 'value') else ar_score

        print(f"  ↳ Faithfulness: {f_val}")
        print(f"  ↳ Answer Relevancy: {ar_val}\n")

        results.append({
            "id": item_id,
            "category": category,
            "question": q,
            "faithfulness": f_val,
            "answer_relevancy": ar_val
        })

    # Generate Final Report
    df = pd.DataFrame(results)
    print("\n🏆 Benchmark Evaluation Summary:")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    benchmark_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "evaluation/benchmarks/resume_benchmark.json"
    )

    asyncio.run(run_benchmark_suite(benchmark_file))