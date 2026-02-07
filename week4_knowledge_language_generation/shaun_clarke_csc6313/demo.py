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


def chunk_text(self, collect_content_func: Callable[[str, str], Dict], text: str, source: str, 
               page: Optional[int] = None, row: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Split text into overlapping chunks while preserving metadata.

    :param text: The text content to chunk
    :type text: str
    :param source: Source filename/path
    :type source: str
    :param page: Optional page number (for PDFs)
    :type page: Optional[int]
    :param row: Optional row number (for CSVs)
    :type row: Optional[int]
    :return: List of chunk dictionaries with content and metadata
    :rtype: List[Dict[str, str]]
    """
    # List to hold citionaries of post processed chunks
    chunks: List = []
    # How big each chunk of text is
    chunk_size: int = 200
    # How much text we reuse from the previous chunk so their is an overlap and we do not miss context or have broken sentences
    # Each chunk of text will overlap the previous one by 50 characters
    chunk_overlap: int = 50

    # If the amount of text is smaller than the chunk_size, return a single cunk
    if len(text) <= chunk_size:
        # Creating dictionary of chunked text
        chunk_dict: Dict = collect_content_func(source, text, page, row)
        # adding chunked text dictionary to chunks list
        chunks.append(chunk_dict)
        return chunks
    
    # the step size determines how far the chunking window/range advances through the text
    # Step_size = chunk_size - chunk_overlap, this means the chunking window will always be slightly smaller than the chunk_size
    # So step_size is what enforces the overlap
    step_size: int = chunk_size - chunk_overlap

    # Looping through text to create chunks
    for starting_slice_position in range(0, len(text), step_size):
        # Calculating the ending_slice_position by adding the starting_slice_position to chunk_size
        # This allows the end to be calculated on the fly, while start moves slightly forward on each iteration.
        ending_slice_position: int = starting_slice_position + chunk_size
        # Chunking text by using start and end to slice the text
        chunk_text: str = text[starting_slice_position:ending_slice_position]
        # Creating dictionary of chunked text
        chunk_dict: Dict = collect_content_func(source, chunk_text, page, row)
        # Adding dictionary of chunked text to list
        chunks.append(chunk_dict)
        # If we are at the end of the text then break the loop
        if ending_slice_position >= len(text):
            break
        
    return chunks






        
        

    # Hints:
    # 1. Calculate step size = chunk_size - chunk_overlap
    # 2. Use a loop with range(0, len(text), step_size)
    # 3. For each position, extract text[position:position+chunk_size]
    # 4. Create dict using collect_file_content() for each chunk
    # 5. Include the metadata (source, page, row) in each chunk

    return chunks


test = [1, 2, 3, 4]
print(test)

test.append({
    "one": 1,
    "two": 2
})
print(test)
test.extend({
    "one": 1,
    "two": 2
})

print(test)
