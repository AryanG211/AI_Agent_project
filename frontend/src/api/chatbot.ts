// API functions to communicate with your Python backend
// You'll need to implement these endpoints in your Python backend

const API_BASE_URL = 'http://localhost:8000'; // Adjust this to your Python server URL

export interface ChatResponse {
  answer: string;
  success: boolean;
  error?: string;
}

export interface ConversationStatus {
  active: boolean;
  listening: boolean;
  processing: boolean;
}

// Start the conversation session
export const startConversation = async (): Promise<{ success: boolean; error?: string }> => {
  try {
    const response = await fetch(`${API_BASE_URL}/start-conversation`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    const data = await response.json();
    return { success: response.ok, error: data.error };
  } catch (error) {
    return { success: false, error: 'Failed to connect to backend' };
  }
};

// Stop the conversation session
export const stopConversation = async (): Promise<{ success: boolean; error?: string }> => {
  try {
    const response = await fetch(`${API_BASE_URL}/stop-conversation`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    const data = await response.json();
    return { success: response.ok, error: data.error };
  } catch (error) {
    return { success: false, error: 'Failed to connect to backend' };
  }
};

// Get conversation status
export const getConversationStatus = async (): Promise<ConversationStatus | null> => {
  try {
    const response = await fetch(`${API_BASE_URL}/status`);
    
    if (response.ok) {
      return await response.json();
    }
    return null;
  } catch (error) {
    return null;
  }
};

// Send a text message (for testing purposes)
export const sendMessage = async (message: string): Promise<ChatResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message }),
    });
    
    const data = await response.json();
    return {
      answer: data.answer || '',
      success: response.ok,
      error: data.error
    };
  } catch (error) {
    return {
      answer: '',
      success: false,
      error: 'Failed to send message'
    };
  }
};