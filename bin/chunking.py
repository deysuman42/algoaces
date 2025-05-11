from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import Tag
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document


def code_handler(element: Tag) -> str:
    data_lang = element.get("data-lang")
    code_format = f"<code:{data_lang}>{element.get_text()}</code>"

    return code_format


def load_data(data):
    """
    Load an Excel file using UnstructuredExcelLoader.
    """
    # Load the Excel file
    # The mode can be "elements" or "elements_with_text"

    loader = UnstructuredExcelLoader(data, mode="elements")
    docs = loader.load()

    return docs

def chunk_docs_recursive(
    docs: List[Document], 
    chunk_size: int = 5000, 
    chunk_overlap: int = 0,
    sheet_name: Optional[str] = None
) -> List[Document]:
    """
    Chunk the documents using RecursiveCharacterTextSplitter with optimizations
    for tabular financial data.
    
    Args:
        docs: List of documents to chunk
        chunk_size: Maximum size of chunks (in characters)
        chunk_overlap: Amount of overlap between chunks (in characters)
        sheet_name: Optional name of the Excel sheet this data comes from
        
    Returns:
        List of chunked documents
    """
    # Create a splitter with separators optimized for tabular data
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n"],
        keep_separator=False,
        add_start_index=True,
    )
    
    # Split the documents into chunks
    chunks = splitter.split_documents(docs)
    
    # # Enhance metadata for better retrieval
    # for i, chunk in enumerate(chunks):
    #     # Add useful metadata
    #     chunk.metadata.update({
    #         "chunk_id": i,
    #         "total_chunks": len(chunks),
    #         "sheet_name": sheet_name or chunk.metadata.get("sheet_name", "unknown"),
    #     })
    
    return chunks