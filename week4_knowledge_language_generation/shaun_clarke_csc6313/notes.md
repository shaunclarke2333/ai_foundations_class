ow implement each TODO in this order:

### **Phase 1: Document Loading (Start Here)**
1. `DocumentLoader.load_documents()` - Load .txt files
2. `DocumentLoader.chunk_text()` - Split into chunks
3. Test with print statements to see your chunks

### **Phase 2: Vector Storage**
4. `VectorStore.__init__()` - Set up ChromaDB
5. `VectorStore.add_documents()` - Store embeddings
6. `VectorStore.search()` - Query similar chunks
7. Test retrieval independently

### **Phase 3: Generation**
8. `RAGChatbot.load_knowledge_base()` - Connect loader to store
9. `RAGChatbot.retrieve_context()` - Format retrieved chunks
10. `RAGChatbot.generate_response()` - Call HuggingFace API
11. `RAGChatbot.chat()` - Combine retrieval + generation

### **Phase 4: Main Loop**
12. `main()` - Interactive chat interface

---

## 📊 **STEP 6: Create Sample Data**

Create `sample_data/` directory with 2-3 .txt files. Here are some ideas:

**document1.txt** (About ChromaDB):
```
ChromaDB is an open-source embedding database. It is designed to make it easy to build 
applications with embeddings. ChromaDB allows you to store embeddings and their metadata, 
and query them efficiently. It is particularly useful for retrieval-augmented generation systems.
```

**document2.txt** (About RAG):
```
Retrieval-Augmented Generation combines retrieval of external information with language generation.
RAG systems first retrieve relevant documents from a knowledge base, then use those documents
as context for generating responses. This approach reduces hallucinations and allows models
to access up-to-date information.
```

**document3.txt** (About Embeddings):
```
Embeddings are dense vector representations of text. They capture semantic meaning in numerical form.
Similar concepts have similar embedding vectors. The sentence-transformers library provides 
pre-trained models for creating embeddings. The all-MiniLM-L6-v2 model is efficient and effective.