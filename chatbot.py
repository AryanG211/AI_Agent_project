import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") #make sure .env file and place your gemini key

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

os.environ["GOOGLE_API_KEY"] = api_key
print("Gemini API key loaded")

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    chunks = splitter.split_documents(pages)
    return chunks

pdf_path = "RAG_Agent.pdf" #Path
documents = load_pdf(pdf_path)
print(f"Loaded and split into {len(documents)} chunks.")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_db = FAISS.from_documents(documents, embeddings)
print("vector store created.")

from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],  # ✅ not summaries
    template="""
You are a helpful assistant that only answers based on the provided context.
Do not use prior knowledge.

Context:
{context}

Question:
{question}

Only answer from the context. If the answer is not in the context, say "I don't know."
"""
)


llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",  
    temperature=0.3,
    max_output_tokens=1024,
    top_p=0.8,
    top_k=40
)

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Create memory object
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 10}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)



print("QA chain ready.")

while True:
    question = input("You: ")
    if question.lower() in ["exit", "quit"]:
        print("🛑 Exiting chat.")
        break
    if not question.strip():
        print("⚠️ Please enter a question.")
        continue
    try:
        response = qa_chain({"query": question})  
        print("AI:", response["result"])  
    except Exception as e:
        print("❌ Error:", e)

