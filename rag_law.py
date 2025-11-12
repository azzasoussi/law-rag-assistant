import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import ollama


logging.basicConfig(level=logging.INFO)

DOC_PATH = "./data/extrait-code-penal.pdf"
MODEL_NAME = "qwen3:0.6b"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"

def load_pdf(doc_path):
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        return None


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


def create_vector_db(chunks):
    ollama.pull(EMBEDDING_MODEL)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name=VECTOR_STORE_NAME,
    )
    logging.info("Vector database created.")
    return vector_db


def create_retriever(vector_db):
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    logging.info("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    template = """
Vous êtes un expert en droit, spécialisé dans l'application de la loi en France. 
Votre rôle est de répondre aux questions en français en vous basant **uniquement sur le contexte fourni**, sans inventer d'informations supplémentaires. 

Contexte: {context}

Instructions : 
1. Fournissez une réponse claire, **très détaillée** et précise. 
2. Si le contexte contient des références légales (articles, codes, sections), mentionnez-les. 
3. Ne formulez pas de conseils personnels ni d'interprétations hors du contexte. 
4. Si la question ne peut pas être répondue à partir du contexte fourni, répondez : 
   "Désolé, je ne dispose pas d'informations suffisantes pour répondre à cette question."

Question: {question}
Réponse :
"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    )

    logging.info("Chain created successfully.")
    return chain

def setup_rag():
    data = load_pdf(DOC_PATH)
    if data is None:
        raise FileNotFoundError('pdf not found')

    chunks = split_documents(data)
    vector_db = create_vector_db(chunks)
    llm = ChatOllama(model=MODEL_NAME)
    retriever = create_retriever(vector_db)
    chain = create_chain(retriever, llm)

    logging.info("RAG chain initialized successfully.")
    return chain


def answer(question):
    chain = setup_rag()
    response = chain.invoke(input=question)
    return response

"""
def main(question):
    data = load_pdf(DOC_PATH)
    if data is None:
        return
    chunks = split_documents(data)
    vector_db = create_vector_db(chunks)
    llm = ChatOllama(model=MODEL_NAME)
    retriever = create_retriever(vector_db, llm)
    chain = create_chain(retriever, llm)
    res = chain.invoke(input=question)
    print("Response:")
    print(res)


if __name__ == "__main__":
    question = "Qui est-ce qui est coupable de trahison et puni de mort ?"
    main(question)

"""