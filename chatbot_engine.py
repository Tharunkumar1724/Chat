"""
PRODUCTION CHATBOT ENGINE WITH GROQ LLAMA 8B
=============================================
Conversational RAG with image understanding using Groq API
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime

from groq import Groq

from rag_engine import RAGPipeline, RetrievalResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("chatbot_engine")


# =============================================================================
# STATE DEFINITIONS
# =============================================================================

@dataclass
class ChatMessage:
    """Structured chat message"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


# =============================================================================
# CHATBOT ENGINE
# =============================================================================

class ProductionChatbot:
    """
    Production-grade chatbot with:
    - Groq Llama 8B Instant
    - RAG-enhanced responses
    - Image understanding
    - Conversation memory
    """
    
    def __init__(
        self,
        groq_api_key: str,
        rag_pipeline: RAGPipeline,
        model_name: str = "llama-3.1-8b-instant",
        temperature: float = 0.7,
        max_context_chunks: int = 5
    ):
        """
        Initialize chatbot engine
        
        Args:
            groq_api_key: Groq API key
            rag_pipeline: Initialized RAG pipeline
            model_name: Groq model name
            temperature: Response creativity (0-1)
            max_context_chunks: Max chunks to use as context
        """
        self.rag = rag_pipeline
        self.max_context_chunks = max_context_chunks
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize Groq client
        self.client = Groq(api_key=groq_api_key)
        
        # Conversation memory
        self.conversations: Dict[str, List[ChatMessage]] = {}
        
        logger.info(f"Initialized chatbot with model: {model_name}")
    
    def _extract_image_info(self, text: str, metadata: Dict) -> Optional[Dict]:
        """Extract structured image information"""
        import re
        
        # Extract image ID, caption, OCR text
        image_match = re.search(r'Image_(\d+)_(\d+):', text)
        if image_match:
            page, idx = image_match.groups()
            
            # Extract caption
            caption_match = re.search(r'Caption: (.+?)(?:\n|$)', text)
            caption = caption_match.group(1) if caption_match else ""
            
            # Extract OCR text
            ocr_match = re.search(r'OCR: (.+?)(?:\n|$)', text)
            ocr_text = ocr_match.group(1) if ocr_match else ""
            
            return {
                "page": int(page),
                "index": int(idx),
                "caption": caption,
                "ocr_text": ocr_text,
                "image_path": metadata.get("image_path", "")
            }
        
        return None
    
    def _generate_response(self, query: str, chunks: List[Dict], images: List[Dict], conversation_history: List[ChatMessage]) -> str:
        """Generate response using Groq API"""
        
        # Build context summary
        context_parts = []
        
        # Add text chunks
        if chunks:
            context_parts.append("=== RELEVANT DOCUMENT CONTEXT ===")
            for i, chunk in enumerate(chunks[:self.max_context_chunks], 1):
                score_pct = chunk['score'] * 100  # Convert to percentage
                context_parts.append(f"\n[Chunk {i}] (Score: {score_pct:.1f}%)")
                context_parts.append(chunk["text"][:800])  # Limit chunk size
        
        # Add image context
        if images:
            context_parts.append("\n\n=== RELEVANT IMAGES ===")
            for i, img in enumerate(images, 1):
                context_parts.append(f"\n[Image {i}] Page {img['page']}")
                context_parts.append(f"Caption: {img['caption']}")
                if img['ocr_text']:
                    context_parts.append(f"Text in image: {img['ocr_text'][:200]}")
        
        context_summary = "\n".join(context_parts)
        
        # Build system prompt
        system_prompt = """You are an intelligent document assistant with access to a knowledge base.

Your capabilities:
- Answer questions based on retrieved document context
- Explain content from PDFs, Word documents, and images
- Provide accurate, well-sourced responses
- Cite specific chunks when relevant

Guidelines:
1. ALWAYS use the provided context to answer questions
2. If context is insufficient, say so clearly
3. For image-related questions, reference the image captions and OCR text
4. Be concise but thorough
5. Use markdown formatting for readability
6. If asked about specific pages, reference the chunk sources

Context Quality Indicators:
- High scores (>0.8) = highly relevant
- Medium scores (0.5-0.8) = somewhat relevant
- Low scores (<0.5) = marginally relevant"""

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation history (last 5 exchanges)
        for msg in conversation_history[-10:]:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add current query with context
        user_message = f"""**User Question:** {query}

**Retrieved Context:**
{context_summary if context_summary else "No relevant context found."}

**Instructions:** Answer the user's question using the context above. Be specific and cite sources when possible."""

        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=2048
            )
            
            response_text = chat_completion.choices[0].message.content
            logger.info(f"Generated response ({len(response_text)} chars)")
            return response_text
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"I apologize, but I encountered an error generating a response: {str(e)}"
    
    def chat(
        self,
        message: str,
        conversation_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main chat interface
        
        Args:
            message: User message
            conversation_id: Conversation identifier for memory
            metadata: Optional metadata
        
        Returns:
            Response dictionary with answer, sources, and metadata
        """
        timestamp = datetime.now().isoformat()
        
        # Retrieve context from RAG
        try:
            results = self.rag.query(message, top_k=self.max_context_chunks)
            chunks = []
            for result in results:
                chunks.append({
                    "chunk_id": result.chunk_id,
                    "text": result.text,
                    "score": float(result.combined_score),
                    "metadata": result.metadata
                })
            logger.info(f"Retrieved {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            chunks = []
        
        # Extract images from chunks
        images = []
        for chunk in chunks:
            text = chunk["text"]
            metadata_dict = chunk.get("metadata", {})
            if "Image_" in text or metadata_dict.get("content_type") == "image":
                image_info = self._extract_image_info(text, metadata_dict)
                if image_info:
                    images.append(image_info)
        
        # Get conversation history
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        history = self.conversations[conversation_id]
        
        # Generate response
        response_text = self._generate_response(message, chunks, images, history)
        
        # Store in conversation memory
        self.conversations[conversation_id].append(
            ChatMessage(role="user", content=message, timestamp=timestamp)
        )
        self.conversations[conversation_id].append(
            ChatMessage(role="assistant", content=response_text, timestamp=timestamp, metadata={
                "chunks_used": len(chunks),
                "images_used": len(images)
            })
        )
        
        # Build response
        return {
            "answer": response_text,
            "sources": chunks,
            "images": images,
            "conversation_id": conversation_id,
            "timestamp": timestamp,
            "metadata": {
                "chunks_retrieved": len(chunks),
                "images_found": len(images),
                "model": self.model_name
            }
        }
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history"""
        if conversation_id not in self.conversations:
            return []
        
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "metadata": msg.metadata
            }
            for msg in self.conversations[conversation_id]
        ]
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Cleared conversation: {conversation_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chatbot statistics"""
        total_conversations = len(self.conversations)
        total_messages = sum(len(conv) for conv in self.conversations.values())
        
        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_context_chunks": self.max_context_chunks
        }


# =============================================================================
# STANDALONE CHATBOT (for testing)
# =============================================================================

def create_chatbot(
    groq_api_key: str,
    index_path: str = "data/index",
    model_name: str = "llama-3.1-8b-instant"
) -> ProductionChatbot:
    """
    Create chatbot instance with existing RAG index
    
    Args:
        groq_api_key: Groq API key
        index_path: Path to FAISS index
        model_name: Groq model
    
    Returns:
        Initialized chatbot
    """
    from rag_engine import RAGPipeline
    
    # Initialize RAG
    rag = RAGPipeline()
    rag.load_index(index_path)
    
    # Create chatbot
    chatbot = ProductionChatbot(
        groq_api_key=groq_api_key,
        rag_pipeline=rag,
        model_name=model_name
    )
    
    return chatbot


# =============================================================================
# INTERACTIVE CHAT SESSION (for testing)
# =============================================================================

def interactive_chat():
    """Run interactive chat session"""
    import sys
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found in environment")
        sys.exit(1)
    
    # Create chatbot
    print("Initializing chatbot...")
    chatbot = create_chatbot(groq_api_key=api_key)
    
    print("\n" + "="*80)
    print("PRODUCTION CHATBOT - Groq Llama 8B Instant")
    print("="*80)
    print("Commands:")
    print("  /clear  - Clear conversation history")
    print("  /stats  - Show chatbot statistics")
    print("  /exit   - Exit chat")
    print("="*80 + "\n")
    
    conversation_id = "cli_session"
    
    while True:
        try:
            # Get user input
            user_input = input("\n🧑 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input == "/exit":
                print("\nGoodbye! 👋")
                break
            
            elif user_input == "/clear":
                chatbot.clear_conversation(conversation_id)
                print("✅ Conversation cleared")
                continue
            
            elif user_input == "/stats":
                stats = chatbot.get_stats()
                print("\n📊 Chatbot Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            
            # Process chat
            print("\n🤖 Assistant: ", end="", flush=True)
            
            response = chatbot.chat(user_input, conversation_id=conversation_id)
            print(response["answer"])
            
            # Show sources
            if response["sources"]:
                print(f"\n📚 Sources ({len(response['sources'])} chunks):")
                for i, source in enumerate(response["sources"][:3], 1):
                    print(f"  [{i}] Score: {source['score']:.3f} | {source['text'][:100]}...")
            
            # Show images
            if response["images"]:
                print(f"\n🖼️  Images ({len(response['images'])} found):")
                for i, img in enumerate(response["images"], 1):
                    print(f"  [{i}] Page {img['page']}: {img['caption'][:100]}...")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    interactive_chat()
