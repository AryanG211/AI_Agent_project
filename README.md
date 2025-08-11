<h1 align="center">📦 RAG-Based Delivery Agent</h1>

<p align="center">
  <em>Intelligent Delivery Assistant – Search, Understand, Respond.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-%23009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-%23F7DF1E?style=for-the-badge&logo=chainlink&logoColor=black" />
  <img src="https://img.shields.io/badge/FAISS-%2300BFFF?style=for-the-badge&logo=facebook&logoColor=white" />
  <img src="https://img.shields.io/badge/Google%20Gemini-%234285F4?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/React.js-%2361DAFB?style=for-the-badge&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/Vite-%23646CFF?style=for-the-badge&logo=vite&logoColor=white" />
  <img src="https://img.shields.io/badge/TailwindCSS-%2306B6D4?style=for-the-badge&logo=tailwind-css&logoColor=white" />
</p>

---

## 🌐 Overview

The **RAG-Based Delivery Agent** is an AI-powered voice-enabled assistant for delivery-related tasks.  
It combines **Retrieval-Augmented Generation (RAG)** with **multilingual voice interaction** to answer delivery-related queries using a knowledge base of documents (e.g., PDF manuals, delivery guidelines).

This agent:
- Listens to the user via speech recognition  
- Understands their query in Hindi, English, Marathi, or Gujarati  
- Retrieves relevant context from a vector store (FAISS)  
- Generates a natural response using Google Gemini or another LLM  
- Speaks the answer back in the same language

---

## 🚀 Features

- 🎙 **Voice Interaction** – Listen & speak in multiple languages
- 🌍 **Multilingual Support** – Hindi, English, Marathi, Gujarati (auto-detected)
- 📄 **Document-Aware** – Reads PDF documents & answers using RAG
- 🧠 **FAISS Vector Search** – Fast semantic retrieval
- 🤖 **Google Gemini Integration** – For high-quality answers
- ⚡ **FastAPI Backend** – Lightweight and responsive API
- 🔄 **Real-Time Conversation Loop** – Continuous Q&A until stopped

---

## 👨‍💻 Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend Framework** | FastAPI |
| **AI/LLM** | Google Gemini via LangChain |
| **Vector Store** | FAISS |
| **Embeddings** | Google Generative AI embeddings |
| **Speech Recognition** | SpeechRecognition (Google STT API) |
| **Text-to-Speech** | gTTS |
| **File Processing** | PyPDF2 / pdfplumber |
| **Language Detection** | langdetect |

---

## 🛠 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/RAG-Delivery-Agent.git
cd RAG-Delivery-Agent
