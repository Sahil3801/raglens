from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
import os

def ingest_pdf():
    pdf_path = "sample.pdf"

    if not os.path.exists(pdf_path):
        print(f"Error: Could not find {pdf_path}. Please place a PDF in the folder.")
        return

    # 1. Extract text from the PDF
    print("Loading document...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("EXTRACTED TEXT:")
    for doc in documents:
        print(doc.page_content)

    # 2. Chop the text into smaller, readable pieces
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 3. Load the free, local embedding model
    print("Loading HuggingFace Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Save the embedded chunks into a local vector database
    print("Saving chunks to local Qdrant database...")
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        path="./qdrant_db",
        collection_name="my_documents",
    )

    print("Ingestion complete! Your vectors are saved in the 'qdrant_db' folder.")


if __name__ == "__main__":
    ingest_pdf()