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
from typing import List, Dict
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests


class DocumentLoader:

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
    def collect_file_content(self, source: str, content: str) -> Dict[str, str]:
        # Creating a dictionary that holds the file name and content as keys and values
        file_data: Dict = {"content": content, "source": source}

        return file_data

    # This method reads all files from a specified directory
    def load_documents(self, directory: str) -> List[Dict[str, str]]:

        # Document types that cn abe loaded
        doc_types: List = [".txt", ".pdf", ".md", ".doc"]

        # This list will be used to hold the dictionaries that have content and sourc keys
        documents: List = []
        # Creating a path object for the specified directory
        files: List = os.listdir(directory)
        # print(files)
        # Extracting content from the files in the directory
        for file in files:
            print(file)
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
                print(f"This extension cannot be parsed at the moment: {extension}")
                continue
            
            # If this is a .txt file parse it as a text file
            if extension == ".txt":
                try:
                    # Opening the text file for reading
                    with open(file_path, "r", encoding="utf-8") as text_file:
                        # Returning the contents of the text file
                        text_file_content = text_file.read()
                        # Using a heler function to create a dictionary with the file name and content before adding it to the documents list
                        text_data_dict = self.collect_file_content(source=file, content=text_file_content)
                        # Adding the text file dictionary to the documents list
                        documents.append(text_data_dict)
                        print(documents)

                except:
                    pass

