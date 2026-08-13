import sys
import types
# Trick Ragas into bypassing the missing VertexAI module
dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat

import os
import requests
import pandas as pd
from datasets import Dataset
from ragas import evaluate
# FIX 1: Import the metric Classes instead of the deprecated instances
from ragas.metrics import Faithfulness, AnswerRelevancy
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def run_evaluation():
    evaluator_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    evaluator_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # FIX 2: Initialize metrics manually and enforce Groq's n=1 limit
    faithfulness_metric = Faithfulness()
    answer_relevancy_metric = AnswerRelevancy(strictness=1) 

    questions = [
        "What is Sahil's current role and company?",
        "Can you list the certifications Sahil has achieved?",
        "What is the capital of France?" # The hallucination test
    ]

    answers = []
    contexts = []

    print("🤖 Sending questions to FastAPI server...")
    for q in questions:
        try:
            response = requests.post("http://127.0.0.1:8000/chat", json={"query": q}).json()
            answers.append(response["answer"])
            contexts.append(response["sources"])
            print(f"✅ Received answer for: '{q}'")
        except Exception as e:
            print(f"❌ Failed to reach server. Error: {e}")
            return

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts
    }
    dataset = Dataset.from_dict(data)

    print("\n⚖️  Running Ragas Evaluation (this takes a moment)...")
    
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness_metric, answer_relevancy_metric], # Pass the initialized metrics here
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    print("\n🏆 Final Scorecard:")
    print(f"Overall Scores: {result}")
    
    df = result.to_pandas()
    print("\nDetailed Breakdown:")
    
    # FIX 3: Print whatever columns Ragas generated safely to avoid KeyErrors
    print(df)

if __name__ == "__main__":
    run_evaluation()