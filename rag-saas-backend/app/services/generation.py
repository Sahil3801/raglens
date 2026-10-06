from threading import Lock
from typing import List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings

class GenerationService:
    """
    Service responsible for interacting with the LLM (via Groq), 
    formatting retrieved document chunks into context, and generating grounded answers.
    """
    def __init__(self):
        # Initialize the ChatGroq LLM client using settings for model name, temperature, and API key
        self.llm = ChatGroq(
            model_name=settings.GROQ_MODEL,
            temperature=0,  # Temperature set to 0 for deterministic, factual, non-creative responses
            api_key=settings.GROQ_API_KEY
        )
        
        # Define a strict system prompt enforcing document-grounded behavior and anti-hallucination rules
        system_prompt = (
            "You are a strict, document-grounded retrieval assistant.\n"
            "Your ONLY purpose is to answer questions using information explicitly supported by the provided Context.\n"
            "Rules:\n"
            "1. Answer ONLY based on the provided Context. Do not use external assumptions or outside knowledge.\n"
            "2. Each context chunk is labeled with '--- CHUNK FROM [Filename] ---'. Chunks from the same file "
            "belong to the same document, and the file name often tells whose document it is "
            "(for example, 'jane_doe_resume.pdf' is Jane Doe's resume). Never combine facts from different files "
            "about the same person or topic.\n"
            "3. If the answer is stated in the Context, give it. Only if the Context contains no relevant "
            "information, reply ONLY with: 'I cannot answer this based on the provided document.'\n"
            "4. Do not engage in small talk or conversational pleasantries.\n\n"
            "5. If the user asks for a summary, construct the best possible summary using ONLY the provided chunks. "
            "Do not apologize or state what you cannot do; just provide the synthesized information directly.\n\n"
            "Context:\n{context}"
        )
        
        # Build a chat prompt template combining the system instructions (with context placeholder) and the human input query
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

    def generate_answer(self, query: str, documents: List[Document]) -> str:
        """
        Takes a user query and a list of retrieved documents, formats them into a single context string,
        and invokes the LLM chain to produce a grounded response.
        """
        # Step 1: Format and bundle the retrieved documents into a single text block with clear source boundaries
        context_parts = []
        for doc in _group_by_source(documents):
            source_name = doc.metadata.get("source_file", "Unknown Document")
            context_parts.append(
                f"--- CHUNK FROM {source_name} ---\n{doc.page_content}\n--- END CHUNK ---"
            )
        # Join all chunks together with double newlines
        context_text = "\n\n".join(context_parts)

        # Step 2: Create a LangChain LCEL (LangChain Expression Language) pipeline: Prompt -> LLM -> String Output Parser
        chain = self.prompt | self.llm | StrOutputParser()
        
        # Step 3: Invoke the chain with the user query and the compiled context text, returning the raw string output
        return chain.invoke({
            "input": query,
            "context": context_text
        })

def _group_by_source(documents: List[Document]) -> List[Document]:
    """Keep each file's chunks together so the LLM reads one document at a time.

    Files appear in the order of their best-ranked chunk, and chunks keep their
    reranked order within a file; the selected chunks themselves do not change.
    """
    groups = {}
    for doc in documents:
        groups.setdefault(doc.metadata.get("source_file", "Unknown Document"), []).append(doc)
    return [doc for group in groups.values() for doc in group]


_generation_service = None
_generation_service_lock = Lock()


def get_shared_generation_service() -> GenerationService:
    """Shared service, created on first use: the Groq client rejects an empty
    API key, and the server must still start (upload, list, delete) without one."""
    global _generation_service
    if _generation_service is None:
        with _generation_service_lock:
            if _generation_service is None:
                _generation_service = GenerationService()
    return _generation_service


def __getattr__(name):
    # Keeps `from app.services.generation import generation_service` working
    # (evaluation scripts) without creating the client at import time.
    if name == "generation_service":
        return get_shared_generation_service()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")