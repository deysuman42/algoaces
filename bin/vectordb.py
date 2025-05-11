from langchain_chroma import Chroma
from langchain.schema import Document


def get_vector_db(persist_directory: str, embedding_function, collection_name: str):
    """
    Create a Chroma vector database.
    """
    # Create a Chroma vector database
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
        collection_name=collection_name,
    )

   

    return db

def create_document_list(docs):
    """
    Create a list of documents from the input data.
    """
    # Create a list of documents
    documents = []
    for idx, doc in enumerate(docs, start=1):
        # Assuming each doc is a dictionary with keys 'content' and 'page_name'
        document = Document(
            page_content=doc.page_content,  # Replace with the actual key for content
            metadata={"page_name": doc.metadata['page_name']},  # Replace with the actual key for page_name
            id=idx
        )
        documents.append(document)
    return documents


def add_documents_to_vector_db(db, documents):
    """
    Add documents to the vector database.
    """
    # Add documents to the vector database
    db.add_documents(documents)
    # Persist the vector database
    # db.persist()
    # Return the vector database
    return db

def get_vector_db_from_persist_directory(persist_directory: str, embedding_function, collection_name: str):
    """
    Load a Chroma vector database from a persist directory.
    """
    # Load a Chroma vector database from a persist directory
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
        collection_name=collection_name,
    )

    return db