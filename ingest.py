from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

def get_vectorstore(presist_dir: str = "./vectorstore"):
    """Load the existing Chroma vector store"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=presist_dir,
        embedding_function=embeddings,
    )


def format_docs(docs):
    """Format the retrieved docs into a single string."""
    return "nn---nn".join(
        f"Source: {doc.metadata.get('source', 'unknown')}n{doc.page_content}"
        for doc in docs
    )


def create_rag_chain():
    """Build the RAG chain using LCEL"""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions 
based on the provided context. Always cite which source document 
your answer comes from. If the context does not contain enough 
information to answer, say so honestly.
         
Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{question}"),
    ])

    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


if __name__ == "__main__":
    chain = create_rag_chain()
    response = chain.invoke("What is this document about?")
    print(response)