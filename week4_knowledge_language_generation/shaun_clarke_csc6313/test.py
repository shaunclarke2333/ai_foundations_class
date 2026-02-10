from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
from shaun_clarke_csc6313_rag_chat_bot import DocumentLoader
from shaun_clarke_csc6313_rag_chat_bot import VectorStore


if __name__ == "__main__":
    #Test: Loading documents
    loader = DocumentLoader(chunk_size=300, chunk_overlap=50)
    docs = loader.load_documents("./documents_project4/")
    print(f"\n{'='*60}")
    print(f"TEST: Document Loading")
    print(f"{'='*60}")
    print(f"Loaded {len(docs)} document segments")
    
    # Showing count of document types
    txt_docs = sum(1 for d in docs if '.txt' in d['source'])
    pdf_docs = sum(1 for d in docs if '.pdf' in d['source'])
    csv_docs = sum(1 for d in docs if '.csv' in d['source'])
    md_docs = sum(1 for d in docs if '.md' in d['source'])
    
    print(f"  - Text files: {txt_docs}")
    print(f"  - PDF pages: {pdf_docs}")
    print(f"  - CSV rows: {csv_docs}")
    print(f"  - Markdown files: {md_docs}")
    
    #Test: Chunking documents
    print(f"\n{'='*60}")
    print(f"TEST: Text Chunking")
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

    # Test: Vector Storage
    print(f"\n{'='*60}")
    print(f"TEST: Vector Storage & Search")
    print(f"{'='*60}")
    
    # Initializing the vector store object 
    vector_store = VectorStore()
    
    # Adding all chunks to the database
    vector_store.add_documents(all_chunks)
    
    # Testing searches with different queries
    test_queries = [
        "What is ChromaDB?",
        "Tell me about computer hardware",
        "What are HTTP status codes?",
        "Explain embeddings"
    ]
    
    for query in test_queries:
        print(f"\n{'─'*60}")
        print(f"Query: {query}")
        print(f"{'─'*60}")
        
        results = vector_store.query_db(query, num_of_results=2)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Source: {result['source']}")
            if result.get('page'):
                print(f"  Page: {result['page']}")
            if result.get('row') is not None:
                print(f"  Row: {result['row']}")
            print(f"  Content: {result['content'][:150]}...")