import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, MessageCircle, Volume2, VolumeX, Loader2 } from 'lucide-react';

const VoiceChatbot = () => {
  const [isConversationActive, setIsConversationActive] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [messages, setMessages] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [voiceLevel, setVoiceLevel] = useState(0);
  const [isUserSpeaking, setIsUserSpeaking] = useState(false);
  const [smoothedVoiceLevel, setSmoothedVoiceLevel] = useState(0);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const messagesEndRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const microphoneRef = useRef(null);
  const streamRef = useRef(null);
  const animationFrameRef = useRef(null);
  const smoothingFactorRef = useRef(0.8);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Initialize audio context and microphone for voice level detection
  const initializeAudioContext = async () => {
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      
      if (!streamRef.current) {
        streamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
      }
      
      if (!analyserRef.current) {
        analyserRef.current = audioContextRef.current.createAnalyser();
        analyserRef.current.fftSize = 512;
        analyserRef.current.smoothingTimeConstant = 0.3;
      }
      
      if (!microphoneRef.current) {
        microphoneRef.current = audioContextRef.current.createMediaStreamSource(streamRef.current);
        microphoneRef.current.connect(analyserRef.current);
      }
      
      startVoiceLevelDetection();
    } catch (error) {
      console.error('Error initializing audio context:', error);
    }
  };

  // Start detecting voice levels
  const startVoiceLevelDetection = () => {
    if (!analyserRef.current) return;
    
    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const detectVoiceLevel = () => {
      if (!analyserRef.current) {
        setVoiceLevel(0);
        setSmoothedVoiceLevel(0);
        setIsUserSpeaking(false);
        animationFrameRef.current = requestAnimationFrame(detectVoiceLevel);
        return;
      }
      
      analyserRef.current.getByteFrequencyData(dataArray);
      
      // Calculate RMS (Root Mean Square) for better voice detection
      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i] * dataArray[i];
      }
      const rms = Math.sqrt(sum / bufferLength);
      
      // Normalize to 0-1 range with better sensitivity
      const normalizedLevel = Math.min(rms / 50, 1);
      const sensitiveLevel = Math.pow(normalizedLevel, 0.6);
      
      setVoiceLevel(sensitiveLevel);
      
      // Apply smoothing for more natural animation
      setSmoothedVoiceLevel(prev => {
        const smoothingFactor = smoothingFactorRef.current;
        return prev * smoothingFactor + sensitiveLevel * (1 - smoothingFactor);
      });
      
      // Lower threshold for better speech detection
      setIsUserSpeaking(sensitiveLevel > 0.05);
      
      animationFrameRef.current = requestAnimationFrame(detectVoiceLevel);
    };
    
    detectVoiceLevel();
  };

  // Cleanup audio context
  const cleanupAudioContext = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    
    if (microphoneRef.current) {
      microphoneRef.current.disconnect();
      microphoneRef.current = null;
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    
    analyserRef.current = null;
    setVoiceLevel(0);
    setSmoothedVoiceLevel(0);
    setIsUserSpeaking(false);
  };

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      cleanupAudioContext();
    };
  }, []);

  const addMessage = (type, text) => {
    const newMessage = {
      id: Date.now().toString(),
      type,
      text,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const startConversation = async () => {
    try {
      // Set conversation active immediately for instant UI transition
      setIsConversationActive(true);
      setConnectionStatus('connecting');
      
      // Initialize audio context for voice level detection immediately
      await initializeAudioContext();
      
      // Call the actual backend API
      const response = await fetch('http://localhost:8000/start-conversation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      const data = await response.json();
      
      if (data.success) {
        setConnectionStatus('connected');
        setIsListening(true);
        addMessage('ai', 'Conversation started! I\'m listening for your voice input.');
        
        // Start polling for real status updates
        startStatusPolling();
      } else {
        throw new Error(data.error || 'Failed to start conversation');
      }
      
    } catch (error) {
      console.error('Failed to start conversation:', error);
      setConnectionStatus('disconnected');
      setIsConversationActive(false);
      addMessage('ai', `Sorry, I couldn't start the conversation: ${error.message}`);
    }
  };

  const stopConversation = async () => {
    try {
      // Call the actual backend API
      const response = await fetch('http://localhost:8000/stop-conversation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || 'Failed to stop conversation');
      }
      
      setIsConversationActive(false);
      setConnectionStatus('disconnected');
      setIsListening(false);
      setIsProcessing(false);
      
      // Cleanup audio context
      cleanupAudioContext();
      
      addMessage('ai', 'Conversation ended. Thank you!');
    } catch (error) {
      console.error('Failed to stop conversation:', error);
      addMessage('ai', `Error stopping conversation: ${error.message}`);
    }
  };

  // Simulate polling for status updates
  const startStatusPolling = () => {
    let pollCount = 0;
    
    const pollStatus = () => {
      if (!isConversationActive) return;
      
      // Poll the actual backend status
      fetch('http://localhost:8000/status')
        .then(response => response.json())
        .then(data => {
          if (data.active !== undefined) {
            setIsListening(data.listening);
            setIsProcessing(data.processing);
            
            // If backend says conversation is not active, update frontend
            if (!data.active && isConversationActive) {
              setIsConversationActive(false);
              setConnectionStatus('disconnected');
              addMessage('ai', 'Conversation ended by backend.');
            }
          }
        })
        .catch(error => {
          console.error('Status polling error:', error);
          pollCount++;
          
          // If we fail to connect multiple times, assume backend is down
          if (pollCount > 3) {
            setConnectionStatus('disconnected');
            addMessage('ai', 'Lost connection to backend. Please restart the conversation.');
          }
        });
      
      setTimeout(pollStatus, 1000); // Poll every second for real-time updates
    };
    
    setTimeout(pollStatus, 1000);
  };

  const handleToggleConversation = () => {
    startConversation();
  };

  const clearChat = () => {
    setMessages([]);
  };

  // Calculate dynamic scale based on voice level
  const getDynamicScale = () => {
    if (!isListening) {
      return 1.0;
    }
    
    if (!isUserSpeaking) {
      return 1.2;
    }
    
    const minScale = 1.2;
    const maxScale = 2.0;
    const dynamicScale = minScale + (smoothedVoiceLevel * (maxScale - minScale));
    
    const randomFactor = 1 + (Math.random() - 0.5) * 0.1;
    return Math.min(dynamicScale * randomFactor, maxScale);
  };
  
  const getDynamicOpacity = () => {
    if (!isListening) return 0.3;
    if (!isUserSpeaking) return 0.5;
    
    const minOpacity = 0.3;
    const maxOpacity = 0.8;
    return minOpacity + (smoothedVoiceLevel * (maxOpacity - minOpacity));
  };
  
  const getAnimationDuration = () => {
    if (!isListening || !isUserSpeaking) return 2;
    
    const minDuration = 0.3;
    const maxDuration = 1.5;
    return maxDuration - (smoothedVoiceLevel * (maxDuration - minDuration));
  };

  const getAnimationStyle = () => {
    if (isProcessing) {
      return 'processing 1.5s ease-in-out infinite';
    }
    
    if (isListening) {
      if (isUserSpeaking) {
        const duration = getAnimationDuration();
        return `voice-responsive ${duration}s ease-in-out infinite`;
      } else {
        return 'breathe 2s ease-in-out infinite';
      }
    }
    
    return 'none';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-t-2xl shadow-lg p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-3 bg-indigo-100 rounded-full">
                <MessageCircle className="w-6 h-6 text-indigo-600" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-800">Voice AI Assistant</h1>
                <p className="text-gray-600">Intelligent delivery partner</p>
              </div>
            </div>
            
            {/* Connection Status */}
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${
                connectionStatus === 'connected' ? 'bg-green-500' : 
                connectionStatus === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'
              }`} />
              <span className="text-sm text-gray-600 capitalize">{connectionStatus}</span>
            </div>
          </div>
        </div>

        {/* Chat Messages */}
        <div className="bg-white shadow-lg max-h-96 overflow-y-auto">
          <div className="p-6 space-y-4">
            {messages.length === 0 ? (
              <div className="text-center py-12">
                <div className="p-4 bg-gray-100 rounded-full w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                  <MessageCircle className="w-8 h-8 text-gray-400" />
                </div>
                <p className="text-gray-500">No messages yet. Start a conversation to begin!</p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl ${
                      message.type === 'user'
                        ? 'bg-indigo-600 text-white'
                        : 'bg-gray-100 text-gray-800'
                    }`}
                  >
                    <p className="text-sm">{message.text}</p>
                    <p className={`text-xs mt-1 ${
                      message.type === 'user' ? 'text-indigo-200' : 'text-gray-500'
                    }`}>
                      {message.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Status Bar */}
        <div className="bg-white shadow-lg p-4 border-t border-gray-200">
          <div className="flex items-center justify-center space-x-4">
            {isListening && (
              <div className="flex items-center space-x-2 text-green-600">
                <Mic className="w-5 h-5 animate-pulse" />
                <span className="text-sm font-medium">
                  {isUserSpeaking ? 'Listening - Speaking detected' : 'Listening...'}
                </span>
              </div>
            )}
            
            {isProcessing && (
              <div className="flex items-center space-x-2 text-blue-600">
                <Loader2 className="w-5 h-5 animate-spin" />
                <span className="text-sm font-medium">Processing...</span>
              </div>
            )}
            
            {!isListening && !isProcessing && isConversationActive && (
              <div className="flex items-center space-x-2 text-gray-500">
                <Volume2 className="w-5 h-5" />
                <span className="text-sm">Ready to listen</span>
              </div>
            )}
          </div>
        </div>

        {/* Control Panel */}
        <div className="bg-white rounded-b-2xl shadow-lg p-6">
          <div className="flex flex-col items-center justify-center space-y-6">
            {/* Main Button Container - Fixed dimensions with seamless transition */}
            <div className="relative flex items-center justify-center" style={{ minHeight: '140px', width: '100%' }}>
              
              {/* Start Conversation Button - Seamless transition */}
              <div className={`absolute inset-0 flex items-center justify-center transition-all duration-200 ease-out ${
                isConversationActive ? 'opacity-0 scale-90 pointer-events-none' : 'opacity-100 scale-100'
              }`}>
                <button
                  onClick={handleToggleConversation}
                  disabled={connectionStatus === 'connecting'}
                  className={`flex items-center space-x-3 px-8 py-4 rounded-full font-semibold text-lg transition-all duration-150 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed ${
                    connectionStatus === 'connecting'
                      ? 'bg-gray-400 text-white shadow-lg'
                      : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-lg'
                  }`}
                >
                  {connectionStatus === 'connecting' ? (
                    <>
                      <Loader2 className="w-6 h-6 animate-spin" />
                      <span>Connecting...</span>
                    </>
                  ) : (
                    <>
                      <Mic className="w-6 h-6" />
                      <span>Start Conversation</span>
                    </>
                  )}
                </button>
              </div>
              
              {/* Animated Microphone Circle - Instant appearance with smooth animation */}
              <div className={`absolute inset-0 flex items-center justify-center transition-all duration-200 ease-out ${
                isConversationActive ? 'opacity-100 scale-100' : 'opacity-0 scale-110 pointer-events-none'
              }`}>
                {/* Animated Microphone Circle - Positioned slightly upward */}
                <div className="relative transform -translate-y-2">
                  {/* Outer breathing circle */}
                  <div 
                    className="absolute inset-0 rounded-full bg-green-200"
                    style={{
                      animation: getAnimationStyle(),
                      transform: `scale(${getDynamicScale()})`,
                      opacity: getDynamicOpacity(),
                      transition: isListening && isUserSpeaking 
                        ? 'transform 0.1s ease-out, opacity 0.1s ease-out'
                        : 'transform 0.3s ease-out, opacity 0.3s ease-out'
                    }}
                  />
                  
                  {/* Additional pulse rings for enhanced voice response */}
                  {isListening && isUserSpeaking && smoothedVoiceLevel > 0.3 && (
                    <>
                      <div 
                        className="absolute inset-0 rounded-full bg-green-400"
                        style={{
                          transform: `scale(${getDynamicScale() * 1.2})`,
                          opacity: getDynamicOpacity() * 0.3,
                          animation: `voice-pulse ${getAnimationDuration() * 0.8}s ease-in-out infinite`
                        }}
                      />
                      {smoothedVoiceLevel > 0.6 && (
                        <div 
                          className="absolute inset-0 rounded-full bg-green-500"
                          style={{
                            transform: `scale(${getDynamicScale() * 1.4})`,
                            opacity: getDynamicOpacity() * 0.2,
                            animation: `voice-pulse ${getAnimationDuration() * 0.6}s ease-in-out infinite`
                          }}
                        />
                      )}
                    </>
                  )}
                  
                  {/* Inner microphone button - Always green when conversation is active */}
                  <div className={`relative z-10 w-20 h-20 rounded-full flex items-center justify-center transition-all duration-200 ${
                    isConversationActive && isListening 
                      ? isUserSpeaking
                        ? 'bg-green-600 shadow-xl shadow-green-600/70'
                        : 'bg-green-500 shadow-lg shadow-green-500/50'
                      : isProcessing 
                      ? 'bg-green-600 shadow-lg shadow-green-600/50'
                      : 'bg-green-500 shadow-lg shadow-green-500/50'
                  }`}
                  style={{
                    transform: isListening && isUserSpeaking 
                      ? `scale(${1 + smoothedVoiceLevel * 0.1})` 
                      : 'scale(1)',
                    transition: 'transform 0.1s ease-out'
                  }}>
                    {isProcessing ? (
                      <Loader2 className="w-8 h-8 text-white animate-spin" />
                    ) : (
                      <Mic className={`w-8 h-8 text-white ${
                        isUserSpeaking ? 'drop-shadow-lg' : ''
                      }`} 
                      style={{
                        filter: isUserSpeaking 
                          ? `drop-shadow(0 0 ${smoothedVoiceLevel * 10}px rgba(255,255,255,0.8))`
                          : 'none'
                      }} />
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Action Buttons Container - Fixed height to prevent layout shifts */}
            <div className="flex items-center justify-center" style={{ minHeight: '50px' }}>
              <div className={`flex items-center space-x-4 transition-all duration-200 ease-out ${
                isConversationActive ? 'opacity-100 scale-100' : 'opacity-0 scale-90 pointer-events-none'
              }`}>
                {/* Stop Conversation Button */}
                <button
                  onClick={stopConversation}
                  className="flex items-center space-x-3 px-6 py-3 bg-red-500 hover:bg-red-600 text-white rounded-full font-semibold transition-all duration-150 transform hover:scale-105 shadow-lg"
                >
                  <MicOff className="w-5 h-5" />
                  <span>Stop Conversation</span>
                </button>

                {/* Clear Chat Button */}
                {messages.length > 0 && (
                  <button
                    onClick={clearChat}
                    className="px-6 py-3 bg-gray-200 hover:bg-gray-300 text-gray-700 rounded-full font-medium transition-all duration-150 transform hover:scale-105"
                  >
                    Clear Chat
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Instructions */}
          <div className="mt-6 p-4 bg-blue-50 rounded-xl">
            <h3 className="font-semibold text-blue-800 mb-2">How to use:</h3>
            <ul className="text-sm text-blue-700 space-y-1">
              <li>• Click "Start Conversation" to begin voice interaction</li>
              <li>• Speak your questions clearly when the system is listening</li>
              <li>• The AI will respond based on the uploaded document</li>
              <li>• Click "Stop Conversation" to end the session</li>
              <li>• The outer circle grows and shrinks based on your voice volume</li>
              <li>• Speak louder to see more dramatic visual effects</li>
            </ul>
          </div>
        </div>
      </div>
      
      {/* Custom CSS for animations */}
      <style jsx>{`
        @keyframes breathe {
          0%, 100% {
            transform: scale(1.1);
            opacity: 0.7;
          }
          50% {
            transform: scale(1.4);
            opacity: 0.3;
          }
        }
        
        @keyframes processing {
          0%, 100% {
            transform: scale(1.1);
            opacity: 0.6;
          }
          50% {
            transform: scale(1.3);
            opacity: 0.4;
          }
        }
        
        @keyframes voice-responsive {
          0%, 100% {
            transform: scale(var(--voice-scale, 1.2));
          }
          50% {
            transform: scale(calc(var(--voice-scale, 1.2) * 1.1));
          }
        }
        
        @keyframes voice-pulse {
          0% {
            transform: scale(0.8);
            opacity: 0.8;
          }
          50% {
            transform: scale(1);
            opacity: 0.4;
          }
          100% {
            transform: scale(1.2);
            opacity: 0;
          }
        }
        
        @keyframes fadeIn {
          0% { opacity: 0; transform: translateY(10px); }
          100% { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
};

export default VoiceChatbot;