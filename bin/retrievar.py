

def vector_retrievar_with_source(vector_db, query, top_k=2):
    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(query)
    # Extract the sources from the retrieved documents
    sources = set([doc.metadata["page_name"] for doc in retrieved_docs])
    return retrieved_docs, sources

def get_context(retrieved_docs):
    # Extract the context from the retrieved documents
    context = [doc.page_content for doc in retrieved_docs]
    return context