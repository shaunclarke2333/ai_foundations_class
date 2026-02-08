import os
from typing import List, Dict, Optional, Callable
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import pandas as pd
import pymupdf4llm
import numpy as np
from numpy import typing as npt
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

class VectorStore:
    """
    This class:
    - initializes ChromaDB and the embeddingmodel.
    - Stores chunked text as vector embeddings.
    - Handles semantic similarity searches
    - Basically handling crud operations
    """
    def __init__(self, collection_name: str = "rag_documents", persist_directory: str = "./chroma_db") -> None:
        
        # Initializing the sentence transformer embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Initializing the Chroma DB client
        self.db_client: chromadb = chromadb.Client()
        # Creating a collection(like a table in SQL but not a table) that will hold all the embeddings,chunked text and metadata
        # Mental note for me, a collection in chromaDB is like a table in SQL so one row in chroma would have embeddings,chunked text and metadata
        self.collection = self.db_client.get_or_create_collection(name=self.collection_name)

        print(f"VectorStore initialized")
        print(f"  - Model: all-MiniLM-L6-v2 (384 dimensions)")
        print(f"  - Collection: {collection_name}")
        print(f"  - Persist directory: {persist_directory}")

    # This method adds document chunks to the DB
    def add_documents(self, chunks: List[Dict[str, str]]) -> None:
        """
        Adds document chunks to the database
        
        :param chunks: List of chunk dictionaries fromt the DocumentLoader
        :type chunks: List[Dict[str, str]]
        """

        print(f"\nAdding {len(chunks)} chunks to the vector database ...")
        # Using a list comprehension to create a list of just the text content from the list of chunks dictionaries
        texts: List = [chunk["content"] for chunk in chunks]
        # Generating embeddings for all the text extracted from the chunks
        print(f"  Generating embeddins ...")
        embeddings: np.ndarray = self.embedding_model.encode(texts, show_progress_bar=True)
        # Converting the embeddings, wich is an np array output to a list of lists, which is what chromadb is expecting
        embeddings_list = embeddings.tolist() 
        # creating unique IDs for each chunk by looping through the number of chunks and using it as a counter
        chunk_ids: List = [f"chunk_{i}" for i in range(len(chunks))]
        # Getting the metadata together for each chunk
        chunk_metadatas: List = []
        # Looping through the chunks to filter out the None fileds so the metadata is clean
        for chunk in chunks:
            # Creating a metadata dict with source(filename with path).
            metadata: Dict = {
                "source": chunk["source"]
            }
            # Only adding pages that don't have a none value. ChromaDB requires a string format so converting to string as well
            if chunk.get("page") is not None:
                metadata["page"]= str(chunk["page"])
            # Only adding rows that don't have a none value. ChromaDB requires a string format so converting to string as well
            if chunk.get("row") is not None:
                metadata["row"] = str(chunk["row"])
            
            # Adding filtered metadata dict to chunk_metadatas list
            chunk_metadatas.append(metadata)

        






