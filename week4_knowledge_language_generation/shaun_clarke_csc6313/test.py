from typing import List, Dict
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
from shaun_clarke_csc6313_rag_chat_bot import DocumentLoader

documents = DocumentLoader()
documents.load_documents("./documents_project4")

