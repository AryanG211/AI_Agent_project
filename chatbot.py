# Importing the libraries
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# API keys
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")
os.environ["GOOGLE_API_KEY"] = api_key
print("Gemini API key loaded")

# Document Loader
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(pages)
    return chunks

pdf_path = "RAG_Agent.pdf"
documents = load_pdf(pdf_path)
print(f"Loaded and split into {len(documents)} chunks.")

# Create embeddings and FAISS vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_db = FAISS.from_documents(documents, embeddings)
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
print("Vector store and retriever created.")

# Create prompt template
custom_prompt = PromptTemplate.from_template("""
You're an AI assistant helping with delivery instructions based on past chat history and helpful context. 
Respond in a professional tone and don't be too friendly.
Don't say things like "based on the provided text" — just give the answer naturally.
Provide the answer only from the context , if it is out of the context just say "Dont Know"
If you don't know the answer, say so honestly.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}
""")

# Initializing the llm
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0.3,
    max_output_tokens=1024,
    top_p=0.8,
    top_k=40
)

# Creating memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Retriever
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": custom_prompt},
    output_key="answer"  
)

print("QA chain ready.")

# Converstion Loop
while True:
    question = input("You: ")
    if question.lower() in ["exit", "quit"]:
        print("Exiting chat.")
        break
    if not question.strip():
        print("Please enter a question.")
        continue
    try:
        response = qa_chain.invoke({"question": question})
        print("AI:", response["answer"])
    except Exception as e:
        print("Error:", e)
