"""
Name: Shaun Clarke
Course: CSC6313 Ai Foundations
Instructor: Margaret Mulhall
Module: 4
Assignment: The Knowledge-Grounded Assistant (RAG)

I modified my RAG chatbot soit can read PDF, CSV, MD. TXT

In this project, you will build a Retrieval-Augmented Generation (RAG) system using ChromaDB.
You will create a chatbot (run out of the terminal,  Extra Challenge: Add a UI) that can answer questions about a specific,
private dataset by retrieving relevant information before generating a response.
This mirrors the industry standard for reducing hallucinations in AI models by "grounding" them in facts.
"""


import os
from typing import List, Dict, Optional, Callable
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import pymupdf4llm
import numpy as np
from huggingface_hub import InferenceClient

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
        self.db_client: chromadb = chromadb.PersistentClient(path=persist_directory)
        # Creating a collection(like a table in SQL but not a table) that will hold all the embeddings,chunked text and metadata
        # Mental note for me, a collection in chromaDB is like a table in SQL so one row in chroma would have embeddings,chunked text and metadata
        self.collection = self.db_client.get_or_create_collection(
            name=collection_name)

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

        # Using a list comprehension to create a list of just the text content from the list of chunks dictionaries
        texts: List = [chunk["content"] for chunk in chunks]
        # Generating embeddings for all the text extracted from the chunks
        embeddings: np.ndarray = self.embedding_model.encode(
            texts, show_progress_bar=False)
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
                metadata["page"] = str(chunk["page"])
            # Only adding rows that don't have a none value. ChromaDB requires a string format so converting to string as well
            if chunk.get("row") is not None:
                metadata["row"] = str(chunk["row"])

            # Adding filtered metadata dict to chunk_metadatas list
            chunk_metadatas.append(metadata)

        # Adding the embeded chunks with their metadata, texts, and chunk_ids to the chromaDB collection
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=chunk_metadatas
        )

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
        documents: List = search_results["documents"][0] if search_results["documents"] else [
        ]
        metadatas: List = search_results["metadatas"][0] if search_results["metadatas"] else [
        ]
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


# This RAGChatbot class is the orchestrator that ties it all together, document loading, vector search, and LLM generation
class RAGChatbot:
    """
    This RAGChatbot class is the orchestrator that ties it all together, document loading, vector search, and LLM generation
    """

    def __init__(self, hf_api_key: str, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Docstring for __init__

        :param hf_api_key: Hugging face api key for authentication.
        :type hf_api_key: str
        :param model_name: The LLM model that will use the retreived chunks and the users question to generate the final answer.
        :type model_name: str
        """
        self.hf_api_key = hf_api_key
        self.model_name = model_name
        self.client = InferenceClient(token=hf_api_key)
        self.document_loader = DocumentLoader(chunk_size=300, chunk_overlap=50)
        self.vector_store = VectorStore()

        print(f"\n{'='*60}")
        print(f"SKYNET CHATBOT INITIALIZED")
        print(f"{'='*60}")
        print(f"model: T1000_{model_name}")
        print(f"Ready to conquer earth, eh em!")
        print(f"I mean ready to load the knowledge base!")

    # This methods loads documents from the directory into chromadb vector database
    def load_knowledge_base(self, directory: str) -> None:
        """
        his methods loads documents from the directory into chromadb vector database

        :param directory: Directory path where documents will be loaded from
        :type directory: str
        """
        # Loading documents from directory
        documents = self.document_loader.load_documents(directory)
        # Empty list to hold chunked documents and their metadata as a list of chunked dictionaries
        all_chunks: List = []
        # Looping through all documents and chunking texts
        for document in documents:
            document_chunks: List[Dict[str, str]] = self.document_loader.chunk_text(
                text=document["content"],
                source=document["source"],
                page=document.get("page"),
                row=document.get("row")
            )
            # Adding the chunked dictionary tot he all chunks list
            all_chunks.extend(document_chunks)
        # Adding chunks to the vector store
        self.vector_store.add_documents(all_chunks)
        print(f" SKYNET knowledge base is now online")

    # This method query's the vector store(chromaDB) to retrieve relevant context chunks for a query.
    def query_vector_db(self, query: str,  num_of_results: int = 3) -> str:
        """
        This method query's the vector store(chromaDB) to retrieve relevant context chunks for a query.

        :param query: The user's question
        :type query: str
        :param num_of_results: Number of chunks to be returned tht associstes with the user's question
        :type num_of_results: int
        :return: A formatted context string that combines all the relevatn returned chunks
        :rtype: str
        """
        # Using the user's question to query the vecor DB and specify the
        # number of relevant vectors we want back that corresponds to the user's question
        relevant_chunks: List = self.vector_store.query_db(
            query, num_of_results)

        # Empty List to hold formatted chunk strings aka context parts
        # I say context parts because the chunks add context to the query when passed to the LLM together
        context_parts: list = []
        # Looping throuh the returned chunks to create a list of formatted chunk strings
        for chunk in relevant_chunks:
            # Creating teh source info header string
            source_info: str = f"Source: {chunk['source']}"
            # If the metadata page exists, add it to the header string
            if chunk.get("page") is not None:
                # Adding the page to the string
                source_info += f", Page: {chunk['page']}"
            # If the matadata for row exists, add it to the header string
            if chunk.get("row") is not None:
                source_info += f", Row: {chunk['row']}"

            # Formatting chunk by adding the source_info as a header to the text rtruned in the chunk
            # The source header helps the model understand which text came from which file.
            # This allows the model to reason better
            formatted_chunk = f"[{source_info}]\n{chunk['content']}\n"
            # Adding the formatted chunk with the source header to the context_parts list
            context_parts.append(formatted_chunk)

        # joining all chunks together with a separator
        # I am using a separator because without a separator things get messy real fast.
        # chunk boundaries will disappear and sources are blurred
        # This can make "reasoning" wonky, so the separator makes it useful context
        context: str = "\n---\n".join(context_parts)

        return context

    # This method generates a response using the hugging face API with context retrieved from the vector DB
    def generate_response(self, query: str, context: str) -> str:
        """
        This method generates a response using:
        - The hugging face API with context retrieved from the vector DB
        - and the user's query

        :param query: The user's question
        :type query: str
        :param context: Context that was retrieved from the vectror DB
        :type context: str
        :return: The generated answer from the LLM
        :rtype: str
        """

        # My attempt at Engineering the prompt with guard rails and context behind the scenes
        prompt: str = f"""You are a helpful assistant. Answer the question using ONLY the provided context below.
        IMPORTANT RULES:
        - If the answer is not in the provided cntext, you must say "I dont know based on the provided information"
        - Do not make up information or use knowledge outside the provided context.
        - Be concise and direct, do not ramble on.
        - Cite the source when possible

        CONTEXT:
        {context}

        QUESTION: {query}

        ANSWER:
        """

        try:
            # Using the inference client to make a post request to the hugging face api
            response = self.client.chat_completion(
                 messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query}
                ],
                model=self.model_name,
                # This limits the response length
                max_tokens=250,
                # This controls how random vs deterministic the output will be, 0.7 is balanced
                temperature=0.7,
                # Nucleus sampling limits tokens to smallest probability mass, lower is safer and more factual.
                # Basically, only consider the most likely words whose combined probability adds up to whatever number we set.
                top_p=0.9,
                # # This setting only returns the generated text and not the promt.
                # return_full_text=False
            )

            # Getting the generated text from teh response
            generated_text: str = response.choices[0].message.content

            return generated_text

        except Exception as e:
            # Saving error message:
            error_msg = str(e)

            # Checking for th eissues i ran into during buildign and testing along with common issues referenced in the documentation
            if "403" in error_msg or "not have access" in error_msg.lower():
                return f"Error: You need to accept the Llama license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct. Details: {error_msg}"
        
            elif "503" in error_msg or "loading" in error_msg.lower():
                return "Error: Model is loading. Please wait 30 seconds and try again."
            
            elif "429" in error_msg or "rate" in error_msg.lower():
                return "Error: Rate limited. Please wait a minute and try again."
            
            else:
                return f"Error generating response: {error_msg}"

    # This method will be the main chat function that will run the full RAG workflow/pipeline
    def chatbot(self, query: str) -> str:
        """
        This method will be the main chat function that will run the full RAG workflow/pipeline

        :param query: The users questions
        :type query: str
        :return: The final answer generated by the model
        :rtype: str
        """
        # Using the user's question(query) to retrieve additional context from vector DB
        context: str = self.query_vector_db(query)
        answer: str = self.generate_response(query, context)

        return answer


# This main function runs the RAG chatbot
def main():

    print("\n" + "="*60)
    print("T1000 CHATBOT - Knowledge Grounded Assistant")
    print("="*60)
    print("\nWelcome! This chatbot answers questions using your documents.")
    print("It also promises not travel back in time to find Sarah Connor.\n")

    # Getting hugging face api key from user input and removing white space
    hf_api_key: str = input(
        "Enter your Hugging Face API key, or else: ").strip()
    # If the user did not enter an api key let hte user know it is required
    if not hf_api_key:
        print(f"Your API key is required")
        return

    # Initializing the rag chatbot
    chatbot = RAGChatbot(hf_api_key)

    # # Getting the documents directory from the user, but adding default value just in case
    # data_dir: str = input(
    #     "Enter your documents directory. [Press enter to use './documents_project4/']").strip()
    # # If the user did not enter a directory, use the default
    # if not data_dir:
    data_dir = "./documents_project4/"

    # Chciking if the directory exists
    if not os.path.exists(data_dir):
        # if the directory doesnt exist, let the user know
        print(f"Directory '{data_dir}' not found!")
        return
    # Creating vector embeddings for the documents and loading them into the vector db
    chatbot.load_knowledge_base(data_dir)

    # Creating an interactive loop
    print("\n" + "="*60)
    print("CHAT STARTED")
    print("="*60)
    print("Ask questions about your documents!")
    print("I am able to read PDFs, markdown, csv and .txt files.\n")
    print("Type 'quit', 'exit', or 'q' to end the conversation.\n")

    while True:
        # Getting the  user's question
        user_question = input("\nYou: ").strip()

        # Making sure the user did not enter an exit command
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("\nThanks for chatting! Oh, tell John Connot i'll see him around\n")
            break

        # If the user's input is empty, restart the loop and ask again
        if not user_question:
            print("Please ask T1000 a question.")
            continue

        # Getting response from the chat bot
        answer = chatbot.chatbot(user_question)

        # outputting the chat bot's response
        print(f"\nT1000: {answer}")


if __name__ == "__main__":
    main()
