import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, Trash2, Image as ImageIcon, FileText } from 'lucide-react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

const ChatSection = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [conversationId] = useState(() => `conv_${Date.now()}`);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post(`${API_URL}/api/chat`, {
        message: input,
        conversation_id: conversationId,
      });

      const assistantMessage = {
        role: 'assistant',
        content: response.data.answer,
        sources: response.data.sources,
        images: response.data.images,
        timestamp: response.data.timestamp,
        metadata: response.data.metadata,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = {
        role: 'assistant',
        content: '❌ Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    if (!window.confirm('Clear conversation history?')) return;

    try {
      await axios.delete(`${API_URL}/api/chat/history/${conversationId}`);
      setMessages([]);
    } catch (error) {
      console.error('Clear error:', error);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-section">
      <div className="chat-header">
        <div className="flex items-center gap-2">
          <Bot className="w-6 h-6 text-blue-500" />
          <h2 className="text-xl font-bold">AI Chat Assistant</h2>
        </div>
        <button
          onClick={handleClear}
          className="clear-btn"
          disabled={messages.length === 0}
          title="Clear conversation"
        >
          <Trash2 className="w-4 h-4" />
          Clear
        </button>
      </div>

      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="empty-state">
            <Bot className="w-16 h-16 text-gray-300 mb-4" />
            <h3 className="text-lg font-semibold text-gray-600 mb-2">
              Start a conversation
            </h3>
            <p className="text-gray-500 text-sm">
              Ask questions about your uploaded documents
            </p>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-icon">
                {msg.role === 'user' ? (
                  <User className="w-5 h-5" />
                ) : (
                  <Bot className="w-5 h-5" />
                )}
              </div>
              <div className="message-content">
                <div className="message-text">{msg.content}</div>
                
                {/* Sources */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources">
                    <div className="sources-header">
                      <FileText className="w-4 h-4" />
                      <span>{msg.sources.length} Source{msg.sources.length > 1 ? 's' : ''}</span>
                    </div>
                    <div className="sources-list">
                      {msg.sources.slice(0, 3).map((source, i) => (
                        <div key={i} className="source-item">
                          <div className="source-score">
                            {(source.score * 100).toFixed(0)}%
                          </div>
                          <div className="source-text">
                            {source.text.substring(0, 150)}...
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Images */}
                {msg.images && msg.images.length > 0 && (
                  <div className="images">
                    <div className="images-header">
                      <ImageIcon className="w-4 h-4" />
                      <span>{msg.images.length} Image{msg.images.length > 1 ? 's' : ''}</span>
                    </div>
                    <div className="images-list">
                      {msg.images.map((img, i) => (
                        <div key={i} className="image-item">
                          <div className="image-info">
                            <span className="image-page">Page {img.page}</span>
                            <span className="image-caption">{img.caption.substring(0, 100)}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Metadata */}
                {msg.metadata && (
                  <div className="metadata">
                    <span className="metadata-item">
                      {msg.metadata.chunks_retrieved} chunks
                    </span>
                    {msg.metadata.images_found > 0 && (
                      <span className="metadata-item">
                        {msg.metadata.images_found} images
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        
        {loading && (
          <div className="message assistant loading">
            <div className="message-icon">
              <Loader2 className="w-5 h-5 animate-spin" />
            </div>
            <div className="message-content">
              <div className="message-text">Thinking...</div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-container">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask a question about your documents..."
          className="chat-input"
          rows={1}
          disabled={loading}
        />
        <button
          onClick={handleSend}
          disabled={!input.trim() || loading}
          className="send-btn"
        >
          {loading ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Send className="w-5 h-5" />
          )}
        </button>
      </div>
    </div>
  );
};

export default ChatSection;
