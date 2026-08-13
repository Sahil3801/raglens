from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware # <-- NEW IMPORT
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader # <-- NEW
from langchain_text_splitters import RecursiveCharacterTextSplitter # <-- NEW
from langchain_core.output_parsers import StrOutputParser
import os
import shutil
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

app = FastAPI(title="Evaluation-First RAG API")

# <-- NEW: CORS Configuration to allow Vite frontend to talk to FastAPI -->
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

# 2. Connect to our local Qdrant database
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="my_documents",
    path="./qdrant_db",
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 3. Initialize the Groq LLM 
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,
)

# 4. Set up the Prompt
system_prompt = (
    "You are a strict document-retrieval assistant. "
    "Your ONLY purpose is to answer questions using information explicitly supported by the provided Context. "
    "Do not use your outside knowledge, assumptions, or general knowledge. "
    "Do not infer or invent information that is not supported by the Context. "
    "If the answer cannot be determined from the Context, including conversational questions such as 'how are you?', "
    "you MUST reply ONLY with: 'I cannot answer this based on the provided document.' "
    "Do not engage in small talk or conversational pleasantries.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 5. Define the data format for our API requests
class ChatRequest(BaseModel):
    query: str

# 6. Create the API Endpoint using modern LCEL
@app.post("/chat")
async def chat(request: ChatRequest):
    # A. Retrieve the relevant documents from Qdrant
    docs = retriever.invoke(request.query)

    print("===== RETRIEVED DOCUMENTS =====")
    for doc in docs:
        print("-----")
        print(doc.page_content)
    
    # B. Combine the text from the documents into one string
    context_text = "\n\n".join(doc.page_content for doc in docs)
    
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
    # A. Save the uploaded file temporarily to disk
    temp_file_path = f"./{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # B. Load and parse the PDF
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    
    # C. Split the document into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # D. Embed and store the chunks in our existing Qdrant database
    vector_store.add_documents(chunks)
    
    # E. Clean up the temporary file from the server
    os.remove(temp_file_path)
    
    return {"filename": file.filename, "message": "Document successfully uploaded and indexed!"}