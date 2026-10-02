import os
from pathlib import Path
import getpass          #provides a secure, portable way to prompt users for sensitive information—such as passwords or API keys—in command-line interfaces

from langchain_huggingface import HuggingFaceEndpoint   #Used to connect to Hugging Face hosted models.
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatGoogleGenerativeAI, ChatOpenAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader   #Used for loading documents.
from dotenv import load_dotenv, find_dotenv       #used to load "file.txt"

_ = load_dotenv(find_dotenv())  #Loads environment variables from a .env file.
#Retriever (searches documents),LLM (generates answer)
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.      
"""


def set_custom_prompt(custom_prompt_template): #reusable skeleton string (like an email, document, or AI prompt) with static text, while input variables are placeholders within that string that are dynamically replaced with actual data at runtime
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])


def get_gemini_api_key():   #Application Programming Interface( bridge that allows different software systems to communicate and share data with each other)
    return os.getenv("GEMINI_API_KEY")


def get_llm_model(api_key: str):
    return ChatOpenAI(
    model="LongCat-Flash-Thinking-2601",
    # stream_usage=True,
    # temperature=None,
    # max_tokens=None,
    # timeout=None,
    # reasoning_effort="low",
    # max_retries=2,
    api_key=os.getenv("LONGCAT_API_KEY"),  # If you prefer to pass api key in directly
    base_url="https://api.longcat.chat/openai",
    # organization="...",
    # other params...
)


def load_documents(data_dir="data"):      #If no folder is provided, use a folder named:"data"
    docs = []     #will store all loaded documents.
    data_path = Path(data_dir)     #Path converts the folder name into a Path object, making file operations easier and platform independent.
    if not data_path.exists():
        raise SystemExit(f"Error: data directory '{data_dir}' does not exist.")
#Stops the entire program.
#terminates the Python script immediately by raising a built-in exception that cleanups up resources ((automatic operations that release external resources like files, network connections, or database locks when a block of code finishes executing) without printing a stack trace
    for path in sorted(data_path.rglob("*.txt")):    #rglob recursively searches for all text files, and sorted ensures consistent loading order.
        docs.extend(TextLoader(str(path)).load())  
#str:Converts Path object into string.Loads content into LangChain Document objects.
    for path in sorted(data_path.rglob("*.pdf")):
        docs.extend(PyPDFLoader(str(path)).load())
#Adds loaded documents into docs list.Each page often becomes a separate Document object.PyPDFLoader extracts text from PDF files and converts them into LangChain documents.
    if not docs:
        raise SystemExit(f"No documents found in '{data_dir}'. Add .txt or .pdf files.") #prevents the RAG system from running without any source documents.
    return docs
#extend: append() method which adds a single item as a whole, extend() iterates over the collection you pass to it and adds each item individually. 

def build_vector_store(source_docs=None):    #documents --> embeddings and store them in vector database
    if source_docs is None:
        source_docs = load_documents()   #If no documents supplied,load documents automatically. 

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")      #AI cannot directly compare text efficiently.AI cannot directly compare text efficiently.This numeric representation is called an embedding vector.
    #The embedding model converts textual documents into dense numerical vectors that capture semantic meaning.
    return Chroma.from_documents(         #Creates a Chroma vector database.
        source_docs,   #Documents to store.
        embedding_model,     #Used to generate vectors.
        collection_name="chatbot",    #Database collection name.
        persist_directory="./chroma_db",
    ) #Without persistence:Program closes,Database disappears


def get_qa_chain():      #searches through a database of documents and uses a Language Model to provide accurate, context-aware answers to user queries.This function creates the complete RAG pipeline by connecting the LLM, retriever, vector store, and custom prompt.
    api_key = get_gemini_api_key()       #Without it:No authentication,No Gemini access
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY must be set in the environment.")

    llm_model = get_llm_model(api_key) #This line initializes the Gemini language model using the API key.
    vector_store = build_vector_store() #This line loads documents, generates embeddings, and creates the Chroma vector database.
    return RetrievalQA.from_chain_type(        #It combines:Retriever + LLM + Prompt into one chain.
        llm=llm_model,
        chain_type="stuff",    # It takes all the text chunks returned by your retriever, pastes (stuffs) them together into a single string, and sends them all at once to the Large Language Model (LLM) as context.
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  #Converts Chroma into a retriever object.Vector database stores information.Retriever searches information.k=3 means the retriever returns the three most relevant document chunks for a query. Cosine Similarity is used
        return_source_documents=True,    #Return answer AND retrieved documents(answer with sources mentioned)
        chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},    #The custom prompt defines how retrieved context and user queries are presented to the language model.
    )


def answer_query(query: str, qa_chain=None):
    if qa_chain is None:
        qa_chain = get_qa_chain()
    return qa_chain.invoke({"query": query})


if __name__ == "__main__":       #Run only when file is executed directly
    api_key = get_gemini_api_key()
    if not api_key:
        api_key = getpass.getpass("Enter your Google AI API key: ")
        os.environ["GEMINI_API_KEY"] = api_key

    qa_chain = get_qa_chain()
    user_query = input("Write Query Here: ")
    response = qa_chain.invoke({"query": user_query})     #to execute or "call" a specific function or block of code
    print("RESULT: ", response["result"])
    print("SOURCE DOCUMENTS: ", response["source_documents"])
