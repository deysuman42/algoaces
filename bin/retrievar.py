from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

def vector_retrievar_with_source(vector_db, query, top_k=2):
    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(query)
    # Extract the sources from the retrieved documents
    sources = set([doc.metadata["page_name"] for doc in retrieved_docs])

    return retrieved_docs, sources

def ensemble_retriever(docs, vector_db, query, top_k=2):

    # Dense retrievar
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": top_k})

    # initialize the bm25 retriever and faiss retriever
    bm25_retriever = BM25Retriever.from_documents(
        docs
    )
    bm25_retriever.k = top_k
    # Extract the sources from the retrieved documents

    # initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
    )

    retrieved_docs = ensemble_retriever.invoke(query)
    sources = set([doc.metadata["page_name"] for doc in retrieved_docs])
    return retrieved_docs, sources


def get_context(retrieved_docs):
    # Extract the context from the retrieved documents
    context = [doc.page_content for doc in retrieved_docs]
    return context