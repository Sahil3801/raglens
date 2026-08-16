from typing import List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings

class GenerationService:
    def __init__(self):
        self.llm = ChatGroq(
            model_name=settings.GROQ_MODEL,
            temperature=0,
            api_key=settings.GROQ_API_KEY
        )
        
        system_prompt = (
            "You are a strict, document-grounded retrieval assistant.\n"
            "Your ONLY purpose is to answer questions using information explicitly supported by the provided Context.\n"
            "Rules:\n"
            "1. Answer ONLY based on the provided Context. Do not use external assumptions or outside knowledge.\n"
            "2. Each context chunk is labeled with '--- CHUNK FROM [Filename] ---'. Respect the source boundary.\n"
            "3. If the context does not contain enough information to answer the question with certainty, "
            "you MUST reply ONLY with: 'I cannot answer this based on the provided document.'\n"
            "4. Do not engage in small talk or conversational pleasantries.\n\n"
            "5. If the user asks for a summary, construct the best possible summary using ONLY the provided chunks. "
            "Do not apologize or state what you cannot do; just provide the synthesized information directly.\n\n"
            "Context:\n{context}"

        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

    def generate_answer(self, query: str, documents: List[Document]) -> str:
        context_parts = []
        for doc in documents:
            source_name = doc.metadata.get("source_file", "Unknown Document")
            context_parts.append(
                f"--- CHUNK FROM {source_name} ---\n{doc.page_content}\n--- END CHUNK ---"
            )
        context_text = "\n\n".join(context_parts)

        chain = self.prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "input": query,
            "context": context_text
        })

generation_service = GenerationService()