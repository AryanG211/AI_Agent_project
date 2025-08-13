import os
import tempfile
import threading
import time
from dotenv import load_dotenv
import speech_recognition as sr
from langdetect import detect
from gtts import gTTS
from playsound import playsound

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# FastAPI imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# FastAPI app setup
app = FastAPI(title="Voice AI Assistant", description="AI Assistant with voice interaction")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    success: bool
    answer: str = ""
    error: str = ""

class ConversationStatus(BaseModel):
    active: bool
    listening: bool
    processing: bool

class ApiResponse(BaseModel):
    success: bool
    message: str = ""
    error: str = ""

# Global state
conversation_active = False
is_listening = False
is_processing = False
conversation_thread = None

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")
os.environ["GOOGLE_API_KEY"] = api_key
print("Gemini API key loaded")

# Load PDF into FAISS
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(pages)
    return chunks

pdf_path = "RAG_Document.pdf"
documents = load_pdf(pdf_path)
print(f"Loaded and split into {len(documents)} chunks.")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_db = FAISS.from_documents(documents, embeddings)
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
print("Vector store created.")

# Prompt templates
prompt_en = PromptTemplate.from_template("""
You're an AI assistant helping with delivery instructions based on past chat history and helpful context. 
Respond in a professional tone and don't be too friendly.
Don't say things like "based on the provided text" — just give the answer naturally.
Provide the answer only from the context, if it is out of the context just say "I Don't Know".
If you don't know the answer, say so honestly. 
Answer naturally in English.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}
""")

prompt_hi = PromptTemplate.from_template("""
You're an AI assistant helping with delivery instructions based on past chat history and helpful context. 
Respond in a professional tone and don't be too friendly.
Don't say things like "based on the provided text" — just give the answer naturally.
Provide the answer only from the context, if it is out of the context just say "Don't Know".
If you don't know the answer, say so honestly.
Answer naturally in Hindi.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}
""")

# LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0.3,
    max_output_tokens=1024,
    top_p=0.8,
    top_k=40
)

# Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# QA Chains for English and Hindi
qa_chain_en = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_en},
    output_key="answer"
)

qa_chain_hi = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_hi},
    output_key="answer"
)

# Speech recognizer
recognizer = sr.Recognizer()
mic = sr.Microphone()

# Normalize language code for TTS
def normalize_lang_for_response(detected_lang):
    if detected_lang.startswith("en"):
        return "en"
    else:
        return "hi"

# Listen and transcribe speech
def listen_and_transcribe():
    global is_listening
    with mic as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        is_listening = True
        audio = recognizer.listen(source)
        is_listening = False
    try:
        text = recognizer.recognize_google(audio)  # auto language detection
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand.")
        return None
    except sr.RequestError as e:
        print(f"Speech Recognition error: {e}")
        return None

# Speak text with gTTS in proper language
def speak_with_gtts(text, lang_code):
    try:
        lang_code = normalize_lang_for_response(lang_code)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            temp_filename = tmp_file.name
        tts = gTTS(text=text, lang=lang_code, tld="co.in", slow=False)
        tts.save(temp_filename)
        playsound(temp_filename)
        os.remove(temp_filename)
    except Exception as e:
        print(f"TTS error: {e}")

# Conversation loop
def conversation_loop():
    global conversation_active, is_listening, is_processing
    print("Starting conversation loop...")
    speak_with_gtts("Hello! How can I help you?", "en")

    while conversation_active:
        try:
            user_input = listen_and_transcribe()
            if not user_input:
                time.sleep(1)
                continue

            if any(word in user_input.lower() for word in ["exit", "quit", "stop", "bye", "goodbye"]):
                speak_with_gtts("Goodbye! Thank you for using the assistant.", "en")
                break

            is_processing = True

            try:
                detected_lang = detect(user_input)
            except:
                detected_lang = "en"

            print(f"Detected language: {detected_lang}")

            # Choose chain and TTS language based on detected language
            if detected_lang.startswith("en"):
                response = qa_chain_en.invoke({"question": user_input})
                tts_lang = "en"
            else:
                response = qa_chain_hi.invoke({"question": user_input})
                tts_lang = "hi"

            ai_text = response["answer"]
            print("AI:", ai_text)

            speak_with_gtts(ai_text, tts_lang)
            is_processing = False

        except Exception as e:
            print(f"Error in conversation loop: {e}")
            is_listening = False
            is_processing = False
            time.sleep(2)

    conversation_active = False
    is_listening = False
    is_processing = False
    print("Conversation loop ended")

# FastAPI routes
@app.get("/", summary="Health Check")
async def root():
    return {"message": "Voice AI Assistant API is running", "chatbot_ready": qa_chain_en is not None and qa_chain_hi is not None}

@app.post("/start-conversation", response_model=ApiResponse)
async def start_conversation():
    global conversation_active, conversation_thread
    if not qa_chain_en or not qa_chain_hi:
        return ApiResponse(success=False, error="Chatbot not initialized.")
    if conversation_active:
        return ApiResponse(success=False, error="Conversation already active.")
    conversation_active = True
    conversation_thread = threading.Thread(target=conversation_loop, daemon=True)
    conversation_thread.start()
    return ApiResponse(success=True, message="Conversation started successfully.")

@app.post("/stop-conversation", response_model=ApiResponse)
async def stop_conversation():
    global conversation_active
    if not conversation_active:
        return ApiResponse(success=False, error="No active conversation.")
    conversation_active = False
    time.sleep(1)
    return ApiResponse(success=True, message="Conversation stopped successfully.")

@app.get("/status", response_model=ConversationStatus)
async def get_status():
    return ConversationStatus(active=conversation_active, listening=is_listening, processing=is_processing)

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    if not qa_chain_en or not qa_chain_hi:
        return ChatResponse(success=False, error="Chatbot not initialized")
    if not message.message:
        return ChatResponse(success=False, error="No message provided")

    try:
        detected_lang = detect(message.message)
    except:
        detected_lang = "en"

    print(f"Detected language: {detected_lang}")

    if detected_lang.startswith("en"):
        response = qa_chain_en.invoke({"question": message.message})
        tts_lang = "en"
    else:
        response = qa_chain_hi.invoke({"question": message.message})
        tts_lang = "hi"

    return ChatResponse(success=True, answer=response["answer"])

# Run FastAPI
if __name__ == "_main_":
    import uvicorn
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
