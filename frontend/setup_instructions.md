# Voice AI Assistant Setup Instructions

## Prerequisites
- Python 3.8 or higher
- Node.js 16 or higher
- Google API Key for Gemini AI
- A PDF document for the RAG system

## Backend Setup

### 1. Navigate to Backend Directory
```bash
cd backend
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and add your Google API key:
   ```
   GOOGLE_API_KEY=your_actual_google_api_key_here
   ```

### 5. Add Your PDF Document
Place your PDF document in the `backend` directory and name it `RAG Agent.pdf`, or update the path in `fastapi_server.py`.

### 6. Start the Backend Server
```bash
python fastapi_server.py
```

Or using uvicorn directly:
```bash
uvicorn fastapi_server:app --host 0.0.0.0 --port 8000 --reload
```

The backend will be available at: http://localhost:8000

## Frontend Setup

### 1. Navigate to Project Root
```bash
cd ..  # Go back to project root if you're in backend directory
```

### 2. Install Frontend Dependencies
```bash
npm install
```

### 3. Start the Frontend Development Server
```bash
npm run dev
```

The frontend will be available at: http://localhost:5173

## Verification

### 1. Check Backend Health
Visit: http://localhost:8000
You should see: `{"message": "Voice AI Assistant API is running", "chatbot_ready": true, "status": "healthy"}`

### 2. Check API Documentation
Visit: http://localhost:8000/docs
This will show the interactive API documentation.

### 3. Test the Frontend
1. Open http://localhost:5173 in your browser
2. Click "Start Conversation"
3. Allow microphone permissions when prompted
4. Speak your question clearly
5. The AI should respond based on your PDF document

## Troubleshooting

### Backend Issues
- **API Key Error**: Make sure your Google API key is valid and has Gemini AI access
- **PDF Not Found**: Ensure your PDF file is in the correct location
- **Speech Recognition**: Make sure you have a working microphone
- **Dependencies**: Try reinstalling requirements: `pip install -r requirements.txt --force-reinstall`

### Frontend Issues
- **CORS Errors**: Make sure the backend is running on port 8000
- **Connection Failed**: Check that both frontend and backend are running
- **Microphone Issues**: Allow microphone permissions in your browser

### Common Solutions
1. **Port Conflicts**: If port 8000 is busy, change it in `fastapi_server.py` and update the frontend API URL
2. **Virtual Environment**: Always activate your virtual environment before running the backend
3. **Firewall**: Make sure your firewall allows connections on ports 8000 and 5173

## Features
- ✅ Voice input in Hindi and English
- ✅ Text-to-speech responses
- ✅ Document-based AI responses (RAG)
- ✅ Real-time conversation status
- ✅ Interactive web interface
- ✅ Voice level visualization

## API Endpoints
- `GET /` - Health check
- `POST /start-conversation` - Start voice conversation
- `POST /stop-conversation` - Stop voice conversation
- `GET /status` - Get conversation status
- `POST /chat` - Send text message (for testing)

## Need Help?
1. Check the console logs in both frontend and backend
2. Visit the API docs at http://localhost:8000/docs
3. Make sure all dependencies are installed correctly
4. Verify your Google API key has the necessary permissions