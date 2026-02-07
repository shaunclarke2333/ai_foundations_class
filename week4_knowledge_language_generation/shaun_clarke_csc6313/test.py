from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
from shaun_clarke_csc6313_rag_chat_bot import DocumentLoader


if __name__ == "__main__":
    # Phase 1 Test: Load documents
    loader = DocumentLoader(chunk_size=300, chunk_overlap=50)
    docs = loader.load_documents("./documents_project4/")
    print(f"\n{'='*60}")
    print(f"PHASE 1 TEST: Document Loading")
    print(f"{'='*60}")
    print(f"Loaded {len(docs)} document segments")
    
    # Show document types
    txt_docs = sum(1 for d in docs if '.txt' in d['source'])
    pdf_docs = sum(1 for d in docs if '.pdf' in d['source'])
    csv_docs = sum(1 for d in docs if '.csv' in d['source'])
    md_docs = sum(1 for d in docs if '.md' in d['source'])
    
    print(f"  - Text files: {txt_docs}")
    print(f"  - PDF pages: {pdf_docs}")
    print(f"  - CSV rows: {csv_docs}")
    print(f"  - Markdown files: {md_docs}")
    
    # Phase 2 Test: Chunk documents
    print(f"\n{'='*60}")
    print(f"PHASE 2 TEST: Text Chunking")
    print(f"{'='*60}")
    
    all_chunks = []
    for doc in docs:
        doc_chunks = loader.chunk_text(
            text=doc['content'],
            source=doc['source'],
            page=doc.get('page'),
            row=doc.get('row')
        )
        all_chunks.extend(doc_chunks)
    
    print(f"Created {len(all_chunks)} total chunks")
    print(f"\nSample chunk:")
    print(f"  Source: {all_chunks[0]['source']}")
    if all_chunks[0].get('page'):
        print(f"  Page: {all_chunks[0]['page']}")
    if all_chunks[0].get('row') is not None:
        print(f"  Row: {all_chunks[0]['row']}")
    print(f"  Content preview: {all_chunks[0]['content'][:150]}...")