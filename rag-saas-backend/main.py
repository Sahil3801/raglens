from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import models
import os
import shutil
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

app = FastAPI(title="Evaluation-First RAG API")

# CORS Configuration to allow Vite frontend to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite's default port
        "http://127.0.0.1:5173"
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# 2. Connect to our local Qdrant database (Safely!)
# Initialize the client exactly ONCE so we don't lock ourselves out
client = QdrantClient(path="./qdrant_db")

# If the collection is missing, create it manually 
# (all-MiniLM-L6-v2 uses 384 dimensions)
if not client.collection_exists("my_documents"):
    client.create_collection(
        collection_name="my_documents",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# Wrap it in LangChain
vector_store = QdrantVectorStore(
    client=client,
    collection_name="my_documents",
    embedding=embeddings,
)

# Use MMR to force diversity, preventing one document from hogging all the chunks!
retriever = vector_store.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 8, "fetch_k": 20} # Fetches 20 behind the scenes, returns the 8 most diverse
)

# 3. Initialize the Groq LLM 
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,
)

# 4. Set up the Prompt
system_prompt = (
    "You are a strict document-retrieval assistant. "
    "Your ONLY purpose is to answer questions using information explicitly supported by the provided Context. "
    "1. Answer ONLY based on the provided Context. Do not use outside knowledge. "
    "2. Pay close attention to the SOURCE DOCUMENT of each context chunk. "
    "Each chunk starts with '--- CHUNK FROM [Filename] ---'. "
    "If the user asks about a specific person (e.g., 'Sahil') and the retrieved context comes from a document belonging to someone else (e.g., 'Thirumalai_Resume.pdf'), you MUST reply ONLY with: 'I cannot answer this based on the provided document.' "
    "3. If the answer cannot be determined from the Context, you MUST reply ONLY with: 'I cannot answer this based on the provided document.' "
    "Do not engage in small talk.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 5. Define the data format for our API requests
class ChatRequest(BaseModel):
    query: str

# 6. Create the API Endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    # A. Retrieve the relevant documents from Qdrant
    docs = retriever.invoke(request.query)
    print("RETRIEVED CHUNKS:", len(docs))

    for i, doc in enumerate(docs):
         print(f"\n--- CHUNK {i+1} ---")
         print(doc.page_content)
         print("SOURCE:", doc.metadata.get("source_file"))
    
    # B. Combine the text, INJECTING THE FILENAME into every chunk so the LLM knows who it belongs to
    context_parts = []
    for doc in docs:
        source_name = doc.metadata.get("source_file", "Unknown Document")
        context_parts.append(f"--- CHUNK FROM {source_name} ---\n{doc.page_content}\n--- END CHUNK ---")
    
    context_text = "\n\n".join(context_parts)
    
    # C. Create a simple chain: Prompt -> LLM -> Text Output
    chain = prompt | llm | StrOutputParser()
    
    # D. Ask the LLM to generate the answer based on the context
    answer = chain.invoke({
        "input": request.query,
        "context": context_text
    })
    
    # E. Return both the answer and the exact source chunks
    return {
        "answer": answer,
        "sources": [doc.page_content for doc in docs]
    }

# 7. Create the Document Upload Endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    temp_file_path = f"./{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    
    # Attach filename metadata to each page chunk
    for doc in documents:
        doc.metadata["source_file"] = file.filename
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    vector_store.add_documents(chunks)
    os.remove(temp_file_path)
    
    return {"filename": file.filename, "message": "Document successfully uploaded and indexed!"}

# 8. List all uploaded documents
@app.get("/documents")
async def list_documents():
    client = vector_store.client
    collection_name = vector_store.collection_name
    
    # Ask Qdrant to return the 'metadata' wrapper
    records, _ = client.scroll(
        collection_name=collection_name,
        with_payload=["metadata"],
        limit=100
    )
    
    files = set()
    for record in records:
        # Safely check if 'metadata' exists, and then if 'source_file' is inside it
        if record.payload and "metadata" in record.payload and "source_file" in record.payload["metadata"]:
            files.add(record.payload["metadata"]["source_file"])
            
    return {"documents": list(files)}
# 9. Delete a specific document
@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    client = vector_store.client
    collection_name = vector_store.collection_name
    
    client.delete(
        collection_name=collection_name,
        points_selector=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.source_file",
                    match=models.MatchValue(value=filename)
                )
            ]
        )
    )
    return {"message": f"Document {filename} deleted successfully."}

# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware # <-- NEW IMPORT
# from pydantic import BaseModel
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_qdrant import QdrantVectorStore
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.document_loaders import PyPDFLoader # <-- NEW
# from langchain_text_splitters import RecursiveCharacterTextSplitter # <-- NEW
# from langchain_core.output_parsers import StrOutputParser
# import os
# import shutil
# from dotenv import load_dotenv

# # Load API keys from .env
# load_dotenv()

# app = FastAPI(title="Evaluation-First RAG API")

# # <-- NEW: CORS Configuration to allow Vite frontend to talk to FastAPI -->
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:5173",  # Vite's default port
#         "http://127.0.0.1:5173"
#     ], 
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # 1. Load the embedding model
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # 2. Connect to our local Qdrant database (with auto-create)
# if not os.path.exists("./qdrant_db"):
#     vector_store = QdrantVectorStore.from_texts(
#         ["Database initialized."],
#         embeddings,
#         path="./qdrant_db",
#         collection_name="my_documents",
#     )
# else:
#     vector_store = QdrantVectorStore.from_existing_collection(
#         embedding=embeddings,
#         collection_name="my_documents",
#         path="./qdrant_db",
#     )
# retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# # 3. Initialize the Groq LLM 
# llm = ChatGroq(
#     model_name="llama-3.1-8b-instant",
#     temperature=0,
# )

# # 4. Set up the Prompt
# system_prompt = (
#     "You are a strict document-retrieval assistant. "
#     "Your ONLY purpose is to answer questions using information explicitly supported by the provided Context. "
#     "Do not use your outside knowledge, assumptions, or general knowledge. "
#     "Do not infer or invent information that is not supported by the Context. "
#     "Pay close attention to NAMES and ENTITIES. If the user asks about a specific person and the Context is about someone else, you MUST reply ONLY with: 'I cannot answer this based on the provided document.'"
#     "If the answer cannot be determined from the Context, including conversational questions such as 'how are you?', "
#     "you MUST reply ONLY with: 'I cannot answer this based on the provided document.' "
#     "Do not engage in small talk or conversational pleasantries.\n\n"
#     "Context:\n{context}"
# )

# prompt = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     ("human", "{input}"),
# ])

# # 5. Define the data format for our API requests
# class ChatRequest(BaseModel):
#     query: str

# # 6. Create the API Endpoint using modern LCEL
# @app.post("/chat")
# async def chat(request: ChatRequest):
#     # A. Retrieve the relevant documents from Qdrant
#     docs = retriever.invoke(request.query)

#     print("===== RETRIEVED DOCUMENTS =====")
#     for doc in docs:
#         print("-----")
#         print(doc.page_content)
    
#     # B. Combine the text from the documents into one string
#     context_text = "\n\n".join(doc.page_content for doc in docs)
    
#     # C. Create a simple chain: Prompt -> LLM -> Text Output
#     chain = prompt | llm | StrOutputParser()
    
#     # D. Ask the LLM to generate the answer based on the context
#     answer = chain.invoke({
#         "input": request.query,
#         "context": context_text
#     })
    
#     # E. Return both the answer and the exact source chunks
#     return {
#         "answer": answer,
#         "sources": [doc.page_content for doc in docs]
#     }

# # 7. Create the Document Upload Endpoint
# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     temp_file_path = f"./{file.filename}"
#     with open(temp_file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
    
#     loader = PyPDFLoader(temp_file_path)
#     documents = loader.load()
    
#     # Attach filename metadata to each page chunk
#     for doc in documents:
#         doc.metadata["source_file"] = file.filename
    
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_documents(documents)
    
#     vector_store.add_documents(chunks)
#     os.remove(temp_file_path)
    
#     return {"filename": file.filename, "message": "Document successfully uploaded and indexed!"}
#     # A. Save the uploaded file temporarily to disk
#     temp_file_path = f"./{file.filename}"
#     with open(temp_file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
    
#     # B. Load and parse the PDF
#     loader = PyPDFLoader(temp_file_path)
#     documents = loader.load()
    
#     # C. Split the document into manageable chunks
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )
#     chunks = text_splitter.split_documents(documents)
    
#     # D. Embed and store the chunks in our existing Qdrant database
#     vector_store.add_documents(chunks)
    
#     # E. Clean up the temporary file from the server
#     os.remove(temp_file_path)
    
#     return {"filename": file.filename, "message": "Document successfully uploaded and indexed!"}

# from qdrant_client import models

# @app.get("/documents")
# async def list_documents():
#     # Query Qdrant client to find unique source files
#     client = vector_store.client
#     collection_name = vector_store.collection_name
    
#     # Scroll through points to collect source file metadata
#     records, _ = client.scroll(
#         collection_name=collection_name,
#         with_payload=["source_file"],
#         limit=100
#     )
    
#     files = set()
#     for record in records:
#         if record.payload and "source_file" in record.payload:
#             files.add(record.payload["source_file"])
            
#     return {"documents": list(files)}

# @app.delete("/documents/{filename}")
# async def delete_document(filename: str):
#     client = vector_store.client
#     collection_name = vector_store.collection_name
    
#     # Delete points matching the source_file metadata
#     client.delete(
#         collection_name=collection_name,
#         points_selector=models.Filter(
#             must=[
#                 models.FieldCondition(
#                     key="metadata.source_file",
#                     match=models.MatchValue(value=filename)
#                 )
#             ]
#         )
#     )
#     return {"message": f"Document {filename} deleted successfully."}