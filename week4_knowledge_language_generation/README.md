Project #4: The Knowledge-Grounded Assistant (RAG)

In this project, you will build a Retrieval-Augmented Generation (RAG) system using ChromaDB. You will create a chatbot (run out of the terminal,  Extra Challenge: Add a UI) that can answer questions about a specific, private dataset by retrieving relevant information before generating a response. This mirrors the industry standard for reducing hallucinations in AI models by "grounding" them in facts.


Core Requirements
1. Dataset & Directory Setup
    - Required Directory: You must create a folder named documents_project4 in your Current Working Directory (CWD).
    - Embeddings Storage: All vector database files and embeddings must be stored locally in your CWD.
    - Testing Protocol: I will be testing your project using my own .txt files. Ensure your ingestion script is robust enough to process any text files placed in the documents_project4 folder.
    - Added Challenge: For those looking for an extra hurdle, try making your system robust enough to handle PDFs, Markdown files or other file times in addition to txt. 
2. Implementation of Retrieval (The "R")
    - Vector Database: Use ChromaDB to store your data locally.
    - Embeddings: Use the sentence-transformers library (model: all-MiniLM-L6-v2) to vectorize your text chunks.
    - Retriever: Implement a module that converts a user query to a vector and fetches the top 2-3 most relevant "chunks" from your database.
3. Implementation of Generation (The "G")
    - Augmented Input: Create a prompt template that merges the Retrieved Context + User Question.
    - Free Inference: Use the Hugging Face Hub (free tier) to send your augmented prompt to an open-source model (e.g., meta-llama/Llama-3.1-8B-Instruct).
    - DO NOT SHARE YOUR KEY!
    - System should ask the user for their HuggingFace key. Once they enter it, it should ask them what question they want answered.
Prompt Engineering: Apply strict "Guardrails" in your system prompt: “Answer the question using ONLY the provided context. If the answer is not in the context, say you do not know.”