"""
Name: Shaun Clarke
Course: CSC6313 Ai Foundations
Instructor: Margaret Mulhall
Module: 4
Assignment: The Knowledge-Grounded Assistant (RAG)

In this project, you will build a Retrieval-Augmented Generation (RAG) system using ChromaDB.
You will create a chatbot (run out of the terminal,  Extra Challenge: Add a UI) that can answer questions about a specific,
private dataset by retrieving relevant information before generating a response.
This mirrors the industry standard for reducing hallucinations in AI models by "grounding" them in facts.
"""

import os
from typing import List, Dict, Optional, Callable
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import pandas as pd
import pymupdf4llm
import numpy as np
from numpy import typing as npt

# This class loads documents to be chunked
class DocumentLoader:
    """
    This class loads documents, parses the content and stores them as chunks in a list of dictionaries.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Docstring for __init__

        :param chunk_size: Number of characters per chunk
        :type chunk_size: int
        :param chunk_overlap: Number of overlapping characters between chunks
        :type chunk_overlap: int
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # Helper method to create a dictionary with the extracted file content
    def collect_file_content(self, source: str, content: str, page: Optional[str] = None, row: Optional[str] = None) -> Dict[str, str]:
        # Creating a dictionary that holds the file name and content as keys and values
        file_data: Dict = {
            "content": content,
            "source": source,
            "page": page,
            "row": row
        }
        return file_data

    # This method returns content from a text file or markdown file
    def get_text_content(self, file_path: str) -> str:
        """
        Docstring for get_text_content

        :param self: Description
        :param file_path: Description
        :type file_path: str
        :return: Description
        :rtype: str
        """
        try:
            # Opening the text file for reading
            with open(file_path, "r", encoding="utf-8") as text_file:
                # Returning the contents of the text file
                text_file_content = text_file.read()
                # Returning file content
                return text_file_content
        except FileNotFoundError:
            raise (f"{file_path} could not be found")
        except Exception as e:
            raise (f"Seems like we hvave a problem {e}")

    # This method returns content from a pdf file
    def get_pdf_content(self, file_path: str, documents: List, collect_content_func: Callable[[str, str], Dict]) -> None:
        """
        This method gets the contents of a pdf file along with the page num and file name

        :param self: Description
        :param file_path: The location of the file
        :type file_path: str
        :param documents: The list that holds dictionaries with all the content extracted from files
        :type documents: List
        :param collect_content_func: the collect_file_content function that takes the filename and file content as params
        :type collect_content_func: Callable[[str, str], Dict]
        """
        try:
            # Getting PDF metadata that includes content, page number and file path. page_chunks=True makes the metadata available
            pages: List[Dict[str, str]] = pymupdf4llm.to_markdown(
                file_path, page_chunks=True)
            # Looping through the list fo dictionaries to extract the needed info from the PDF metadata
            for page_dict in pages:
                # Extracting the pdf page content
                page_content: str = page_dict["text"]
                # Getting metadata for the specific page so we can get the page number later
                metadata: str = page_dict["metadata"]
                # Extracting number for the specific page
                page_num: int = metadata["page"]
                # filename: str = os.path.basename(metadata["file_path"]) # Stripping path to get only filename

                # creating a dictionary with the needed page details
                pdf_data_dict = collect_content_func(
                    file_path, page_content, page_num)
                # adding page data  to documents list
                documents.append(pdf_data_dict)
        except FileNotFoundError:
            raise (f"{file_path} could not be found")
        except Exception as e:
            raise (f"Seems like we have a problem {e}")

    # This method returns the content from a CSV
    def get_csv_content(self, file_path: str, documents: List, collect_content_func: Callable[[str, str], Dict]) -> None:
        """
        This method gets the contents of a csv file along with the row num and file name

        :param self: Description
        :param file_path: The location of the file
        :type file_path: str
        :param documents: The list that holds dictionaries with all the content extracted from files
        :type documents: List
        :param collect_content_func: the collect_file_content function that takes the filename and file content as params
        :type collect_content_func: Callable[[str, str], Dict]
        """
        try:
            # Reading in CSV data as a dataframe
            df = pd.read_csv(file_path)
            # Looping through the dataframe getting the index and row data
            # the index is the row number
            # # the row is a pandas series that has the column value parirs for that row
            for index, row in df.iterrows():
                # Empty list to hold the formatted column: value strings the specified row
                lines: list = []
                # replacing all missing NaN values in the specifed row with Unknown
                row = row.fillna('Unknown')
                # looping through each column value pair in the specified row
                # column is the column name
                # value is going to be the value for that column in the specified row
                for column, value in row.items():
                    # adding column and value as key: value pairs so the row data makes sense
                    lines.append(f"{column}: {value}")
                # Joining all the column: value lines as one multi line string
                content = "\n".join(lines)

                # Creating a dictionary with the specified row's data.
                csv_data_dict = collect_content_func(
                    file_path, content, None, index)
                # Adding the processed row to the documents list
                documents.append(csv_data_dict)
        except FileNotFoundError:
            raise (f"{file_path} could not be found")
        except Exception as e:
            raise (f"Seems like we have a problem {e}")

    # This method reads all files from a specified directory
    def load_documents(self, directory: str) -> List[Dict[str, str]]:
        """
        Docstring for load_documents

        :param self: Description
        :param directory: Description
        :type directory: str
        :return: Description
        :rtype: List[Dict[str, str]]
        """
        # Document types that cn abe loaded
        doc_types: List = [".txt", ".pdf", ".md", ".csv"]

        # This list will be used to hold the dictionaries that have content and sourc keys
        documents: List = []
        # Creating a path object for the specified directory
        files: List = os.listdir(directory)
        # print(files)
        # Extracting content from the files in the directory
        for file in files:
            # print(file)
            # Creating absolute filepath
            file_path: str = os.path.join(directory, file)
            # Getting file extension by using split, also using a throw away variable for the filename becasue we only need the extension
            _: str
            extension: str
            _, extension = os.path.splitext(file)
            # making sure extension is lower case
            extension: str = extension.lower()

            # If this item is not a file skip it.
            if not os.path.isfile(file_path):
                # print(f"This is a directory, and here is the path: {file_path}")
                continue
            # If the file extension is not in the approved list skip it
            if extension not in doc_types:
                # print(f"This extension cannot be parsed at the moment: {extension}")
                continue

            # If this is a .txt file parse it as a text file
            if extension == ".txt":
                # Getting content from text file
                text_file_content: str = self.get_text_content(file_path)
                # Using a heler function to create a dictionary with the file name and content before adding it to the documents list
                text_data_dict = self.collect_file_content(
                    source=file_path, content=text_file_content)
                # Adding text file content to list
                documents.append(text_data_dict)

            # If this is a PDF file parse it as a PDF file
            if extension == ".pdf":
                # Getting content and metadata from pdf file
                self.get_pdf_content(file_path, documents,
                                     self.collect_file_content)

            # If this is a markdown file .md parse it as such
            if extension == ".md":
                # Getting content from text file using get_text_content function because it works for .md files as well
                markdown_file_content: str = self.get_text_content(file_path)
                # Using a heler function to create a dictionary with the file name and content before adding it to the documents list
                markdown_data_dict = self.collect_file_content(
                    source=file_path, content=markdown_file_content)
                # Adding markdown file content to list
                documents.append(markdown_data_dict)

            # If this is a CSV file parse it as a .CSV
            if extension == ".csv":
                # Getting content, row number and filename from the csv
                self.get_csv_content(file_path, documents,
                                     self.collect_file_content)

        return documents

    #
    def chunk_text(self, text: str, source: str,
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
            chunk_dict: Dict = self.collect_file_content(
                source, text, page, row)
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
            chunk_dict: Dict = self.collect_file_content(
                source, chunk_text, page, row)
            # Adding dictionary of chunked text to list
            chunks.append(chunk_dict)
            # If we are at the end of the text then break the loop
            if ending_slice_position >= len(text):
                break

        return chunks
    
# This class manages Chroma DB and stores chunks as embeddings
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
        self.collection = self.db_client.get_or_create_collection(name=collection_name)

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

        # Adding the embeded chunks with their metadata, texts, and chunk_ids to the chromaDB collection
        print(f" Hold on to your potatoes, slick we are about ot add processed data to chromaDB ...")
        self.collection.add(
            ids = chunk_ids,
            embeddings = embeddings_list,
            documents = texts,
            metadatas = chunk_metadatas
        )

        print(f" Hot dawg we did it. we added {len(chunks)} to chromadDB")

    
    # This method is responsible for querying chromadb
    def query_db(self, query: str, num_of_results: int = 3, ) -> list[Dict[str, str]]:
        """
        This method allows the user to query the DB using the user's question

        :param query: The user's question
        :type query: str
        :param num_of_results: Numbe rof results to return, the default is 3.
        :type num_of_results: int
        :return: Returns the list of relevant chunk dicts that containt the text, source and metadata
        :rtype: list[Dict[str, str]]
        """

        # Cnverting the query to embedding before it can be used to query the vector db
        # Chroma db will expect a vector representation of the query
        # After some research i found out chromadb also does the embedding.
        # But that depends on how it is configured, so to be safe i am doing the embedding separately.
        query_embedding = self.embedding_model.encode(query)
        # Using the vector representation of the query above to query the vector DB
        search_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=num_of_results
        )
        
        # The section below will be formating search results into list of dictionaries(personal opinion) because the raw output returned by chromadb is a wonky list of lists
        # The list  contaits lists of: documents metadatas and similarity scores
        # plus the dictionary matches the the shape that i have already been using for all the other data.
        
        # Empty list to hold dicts
        results: list = []
        # Extracting the list of document and metadata results form the search reults 
        # If documents exists, return it, if not return an empty list so we dont crash.
        documents: List = search_results["documents"][0] if search_results["documents"] else []
        metadatas: List = search_results["metadatas"][0] if search_results["metadatas"] else []
        # Combining the data insdide the extracted list of documents and metadatas into a dictionary so we can add them to the results list
        # data inside the documents and metadatas arethe vector embeddings that was returned from the search along with the metadata.
        for i in range(len(documents)):
            result_dict: Dict = {
                "content": documents[i],
                "source": metadatas[i].get("source", "Unknown"),
                "page": metadatas[i].get("page"),
                "row": metadatas[i].get("row")
            }

            # Adding the result dict to the results list
            results.append(result_dict)

        return results





