from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import Tag


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
    # "elements" will load the file as a list of elements
    # "elements_with_text" will load the file as a list of elements with text
    # "elements" is the default mode
    # "elements_with_text" is the default mode for UnstructuredExcelLoader

    loader = UnstructuredExcelLoader(data, mode="elements")
    docs = loader.load()

    return docs


def chunk_docs_recursive(docs, chunk_size=50, chunk_overlap=0):
    """
    Chunk the documents using RecursiveCharacterTextSplitter.
    """
    # Create a splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? "],
    )

    # Split the documents into chunks
    chunks = splitter.split_documents(docs)

    return chunks
