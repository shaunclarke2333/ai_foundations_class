import os
from typing import List, Dict, Optional, Callable
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import pandas as pd
import pymupdf4llm
from shaun_clarke_csc6313_rag_chat_bot import DocumentLoader

# Helper method to create a dictionary with the extracted file content


def collect_file_content(source: str, content: str, page: Optional[str] = None, row: Optional[str] = None) -> Dict[str, str]:
    # Creating a dictionary that holds the file name and content as keys and values
    file_data: Dict = {
        "content": content,
        "source": source,
        "page": page,
        "row": row
    }
    return file_data


documents = DocumentLoader()
parsed_docs = documents.load_documents("./documents_project4")

# for doc in parsed_docs:

#     # document text
#     content: str = doc["content"]
#     # The document where the text came from
#     source: str = doc["source"]
#     # If relevant the page in the document where this text came from
#     page: int = doc["page"]
#     # If relevant the row in the document where this text came from
#     row: int = doc["row"]

#     print(f"content: {content}")
#     print(f"source: {source}")
#     print(f"page: {page}")
#     print(f"row: {row}\n")


#   Initializing the Chroma DB client
db_client: chromadb = chromadb.Client()

# Creating a collection(like a table in SQL but not a table) that will hold all the embeddings,chunked text and metadata
# Mental note for me, a collection in chromaDB is like a table in SQL so one row in chroma would have embeddings,chunked text and metadata
collection = db_client.get_or_create_collection(name="test_collection")


