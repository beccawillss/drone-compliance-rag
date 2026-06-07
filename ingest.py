from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

def load_documents(docs_dir: str = "./docs"):
    """Load all supported documents from a directory."""
    loaders = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
    }
    documents = []
    for file in os.listdir(docs_dir):
        ext = os.path.splitext(file)[1].lower()
        if ext in loaders:
            loader = loaders[ext](os.path.join(docs_dir, file))
            documents.extend(loader.load())
    print(f"Loaded {len(documents)} documents")
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks for indexing."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["nn", "n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vectorstore(chunks, persist_dir: str = "./vectorstore"):
    """Create a Chroma vector store from document chunks."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"Created vector store with {vectorstore._collection.count()} vectors")
    return vectorstore


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    vectorstore = create_vectorstore(chunks)