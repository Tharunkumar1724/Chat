"""
PRODUCTION RAG ENGINE
======================
Expert-level hybrid chunking with semantic similarity and FAISS HNSW indexing.
Follows official RAG strategy for production-grade retrieval.
"""

import re
import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Document processing
from docx import Document
import pdfplumber
import pytesseract

# Embedding and indexing
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rag_engine")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Chunk:
    """Represents a semantic chunk with metadata"""
    chunk_id: str
    text: str
    source: str
    paragraph_index: int
    sub_chunk_index: int
    token_count: int
    embedding: np.ndarray = None


@dataclass
class RetrievalResult:
    """Result from retrieval pipeline"""
    chunk_id: str
    text: str
    combined_score: float
    embedding_score: float
    rerank_score: float
    metadata: Dict[str, Any]


# =============================================================================
# 1. DOCUMENT LOADING
# =============================================================================

def load_file(filepath: str) -> str:
    """
    Load and extract text from PDF, DOCX, or TXT files.
    
    Args:
        filepath: Path to document
    
    Returns:
        Extracted text as string
    """
    ext = Path(filepath).suffix.lower()
    
    if ext == '.txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    elif ext == '.docx':
        doc = Document(filepath)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return '\n\n'.join(paragraphs)
    
    elif ext == '.pdf':
        text_parts = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return '\n\n'.join(text_parts)
    
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def normalize_text(text: str) -> str:
    """
    Normalize whitespace and remove artifacts.
    
    Args:
        text: Raw text
    
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\n\n+', '\n\n', text)
    
    # Fix broken words (hyphenation)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Remove page numbers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'\nPage \d+( of \d+)?\n', '\n', text, flags=re.IGNORECASE)
    
    return text.strip()


def extract_paragraphs(text: str) -> List[str]:
    """
    Extract paragraphs from text, preserving boundaries.
    
    Args:
        text: Normalized text
    
    Returns:
        List of paragraphs
    """
    # Split by double newlines
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # Filter out very short paragraphs (likely artifacts)
    paragraphs = [p for p in paragraphs if len(p.split()) >= 5]
    
    return paragraphs


# =============================================================================
# 2. TOKENIZATION
# =============================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate token count (rough approximation: 1 token ≈ 4 chars).
    
    Args:
        text: Input text
    
    Returns:
        Estimated token count
    """
    return len(text) // 4


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple regex.
    
    Args:
        text: Input text
    
    Returns:
        List of sentences
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


# =============================================================================
# 3. HYBRID SEMANTIC CHUNKING (CORE ALGORITHM)
# =============================================================================

class SemanticChunker:
    """
    Hybrid chunking with semantic similarity.
    - Base: paragraphs
    - Re-chunk using sentence-level embeddings
    - Target: 250 tokens, Min: 150, Max: 400
    - Overlap: 40-60 tokens
    """
    
    def __init__(
        self,
        embedding_model: SentenceTransformer,
        target_tokens: int = 250,
        min_tokens: int = 150,
        max_tokens: int = 400,
        overlap_tokens: int = 50,
        similarity_threshold: float = 0.60
    ):
        self.embedding_model = embedding_model
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.similarity_threshold = similarity_threshold
    
    def semantic_chunk_paragraph(
        self,
        paragraph: str,
        paragraph_index: int,
        source: str
    ) -> List[Chunk]:
        """
        Chunk a single paragraph using semantic similarity.
        
        Args:
            paragraph: Paragraph text
            paragraph_index: Index of paragraph in document
            source: Source filename
        
        Returns:
            List of semantic chunks
        """
        # Split into sentences
        sentences = split_into_sentences(paragraph)
        
        if not sentences:
            return []
        
        # Embed sentences
        sentence_embeddings = self.embedding_model.encode(
            sentences,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        chunks = []
        current_chunk_sentences = []
        current_chunk_tokens = 0
        sub_chunk_index = 0
        
        for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
            sentence_tokens = estimate_tokens(sentence)
            
            # Check if adding this sentence would exceed max tokens
            if current_chunk_tokens + sentence_tokens > self.max_tokens and current_chunk_sentences:
                # Finalize current chunk
                chunk = self._create_chunk(
                    current_chunk_sentences,
                    source,
                    paragraph_index,
                    sub_chunk_index
                )
                chunks.append(chunk)
                sub_chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_chunk_sentences)
                current_chunk_sentences = [overlap_text] if overlap_text else []
                current_chunk_tokens = estimate_tokens(overlap_text) if overlap_text else 0
            
            # Check semantic similarity (if chunk has content)
            if current_chunk_sentences and i > 0:
                # Calculate similarity between current chunk and new sentence
                chunk_text = ' '.join(current_chunk_sentences)
                chunk_embedding = self.embedding_model.encode(
                    chunk_text,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
                similarity = cosine_similarity(
                    chunk_embedding.reshape(1, -1),
                    embedding.reshape(1, -1)
                )[0][0]
                
                # If similarity drops and we have enough tokens, start new chunk
                if (similarity < self.similarity_threshold and 
                    current_chunk_tokens >= self.min_tokens):
                    chunk = self._create_chunk(
                        current_chunk_sentences,
                        source,
                        paragraph_index,
                        sub_chunk_index
                    )
                    chunks.append(chunk)
                    sub_chunk_index += 1
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap(current_chunk_sentences)
                    current_chunk_sentences = [overlap_text] if overlap_text else []
                    current_chunk_tokens = estimate_tokens(overlap_text) if overlap_text else 0
            
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_tokens += sentence_tokens
            
            # If we've reached target size, consider finalizing
            if current_chunk_tokens >= self.target_tokens:
                chunk = self._create_chunk(
                    current_chunk_sentences,
                    source,
                    paragraph_index,
                    sub_chunk_index
                )
                chunks.append(chunk)
                sub_chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_chunk_sentences)
                current_chunk_sentences = [overlap_text] if overlap_text else []
                current_chunk_tokens = estimate_tokens(overlap_text) if overlap_text else 0
        
        # Finalize last chunk
        if current_chunk_sentences:
            chunk = self._create_chunk(
                current_chunk_sentences,
                source,
                paragraph_index,
                sub_chunk_index
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_overlap(self, sentences: List[str]) -> str:
        """
        Get overlap text (40-60 tokens from end of chunk).
        
        Args:
            sentences: List of sentences in current chunk
        
        Returns:
            Overlap text
        """
        if not sentences:
            return ""
        
        # Take last sentences until we have 40-60 tokens
        overlap_sentences = []
        overlap_tokens = 0
        
        for sentence in reversed(sentences):
            tokens = estimate_tokens(sentence)
            if overlap_tokens + tokens > 60:
                break
            overlap_sentences.insert(0, sentence)
            overlap_tokens += tokens
            if overlap_tokens >= 40:
                break
        
        return ' '.join(overlap_sentences)
    
    def _create_chunk(
        self,
        sentences: List[str],
        source: str,
        paragraph_index: int,
        sub_chunk_index: int
    ) -> Chunk:
        """Create a Chunk object from sentences."""
        text = ' '.join(sentences)
        chunk_id = self._generate_chunk_id(source, paragraph_index, sub_chunk_index)
        
        return Chunk(
            chunk_id=chunk_id,
            text=text,
            source=source,
            paragraph_index=paragraph_index,
            sub_chunk_index=sub_chunk_index,
            token_count=estimate_tokens(text)
        )
    
    def _generate_chunk_id(self, source: str, para_idx: int, sub_idx: int) -> str:
        """Generate unique chunk ID."""
        base = f"{Path(source).stem}_p{para_idx}_s{sub_idx}"
        hash_obj = hashlib.md5(base.encode())
        return f"{base}_{hash_obj.hexdigest()[:8]}"


def create_chunks_from_text(
    text: str,
    source: str,
    embedding_model: SentenceTransformer
) -> List[Chunk]:
    """
    Main chunking pipeline: text -> paragraphs -> semantic chunks.
    
    Args:
        text: Input text
        source: Source filename
        embedding_model: SentenceTransformer model
    
    Returns:
        List of semantic chunks
    """
    # Normalize text
    normalized = normalize_text(text)
    
    # Extract paragraphs
    paragraphs = extract_paragraphs(normalized)
    
    logger.info(f"Extracted {len(paragraphs)} paragraphs from {source}")
    
    # Initialize chunker
    chunker = SemanticChunker(embedding_model)
    
    # Chunk each paragraph
    all_chunks = []
    for i, paragraph in enumerate(paragraphs):
        chunks = chunker.semantic_chunk_paragraph(paragraph, i, source)
        all_chunks.extend(chunks)
    
    logger.info(f"Created {len(all_chunks)} semantic chunks from {source}")
    
    return all_chunks


# =============================================================================
# 4. EMBEDDING
# =============================================================================

def embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Embed texts using SentenceTransformer and normalize.
    
    Args:
        texts: List of texts to embed
        model: SentenceTransformer model
    
    Returns:
        Normalized embeddings (N x D)
    """
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    )
    return embeddings


def embed_chunks(chunks: List[Chunk], model: SentenceTransformer) -> List[Chunk]:
    """
    Embed chunks and attach embeddings.
    
    Args:
        chunks: List of chunks
        model: SentenceTransformer model
    
    Returns:
        Chunks with embeddings attached
    """
    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts, model)
    
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding
    
    return chunks


# =============================================================================
# 5. FAISS HNSW INDEXING
# =============================================================================

class FAISSIndex:
    """
    FAISS HNSW index for efficient ANN search.
    Uses IndexHNSWFlat with inner product (cosine similarity).
    """
    
    def __init__(self, dimension: int, M: int = 32, efConstruction: int = 200):
        """
        Initialize HNSW index.
        
        Args:
            dimension: Embedding dimension
            M: HNSW M parameter (connections per node)
            efConstruction: HNSW construction parameter
        """
        self.dimension = dimension
        self.index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = efConstruction
        self.index.hnsw.efSearch = 64
        
        self.chunks = []
        self.chunk_texts = []
        self.chunk_ids = []
        self.metadata = []
    
    def add_chunks(self, chunks: List[Chunk]):
        """
        Add chunks to index.
        
        Args:
            chunks: List of chunks with embeddings
        """
        # Extract embeddings
        embeddings = np.array([chunk.embedding for chunk in chunks]).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store chunks and metadata
        self.chunks.extend(chunks)
        self.chunk_texts.extend([chunk.text for chunk in chunks])
        self.chunk_ids.extend([chunk.chunk_id for chunk in chunks])
        self.metadata.extend([{
            'source': chunk.source,
            'paragraph_index': chunk.paragraph_index,
            'sub_chunk_index': chunk.sub_chunk_index,
            'token_count': chunk.token_count
        } for chunk in chunks])
        
        logger.info(f"Added {len(chunks)} chunks to FAISS index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search index for nearest neighbors.
        
        Args:
            query_embedding: Query embedding (normalized)
            top_k: Number of results to return
        
        Returns:
            (scores, indices)
        """
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        scores, indices = self.index.search(query_embedding, top_k)
        return scores[0], indices[0]
    
    def set_ef_search(self, ef_search: int):
        """Set efSearch parameter for query time."""
        self.index.hnsw.efSearch = ef_search
    
    def save(self, index_path: str, metadata_path: str):
        """Save index and metadata."""
        faiss.write_index(self.index, index_path)
        
        import pickle
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'chunk_texts': self.chunk_texts,
                'chunk_ids': self.chunk_ids,
                'metadata': self.metadata,
                'dimension': self.dimension
            }, f)
        
        logger.info(f"Saved index to {index_path} and metadata to {metadata_path}")
    
    def load(self, index_path: str, metadata_path: str):
        """Load index and metadata."""
        self.index = faiss.read_index(index_path)
        
        import pickle
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.chunk_texts = data['chunk_texts']
            self.chunk_ids = data['chunk_ids']
            self.metadata = data['metadata']
            self.dimension = data['dimension']
        
        logger.info(f"Loaded index from {index_path} with {self.index.ntotal} vectors")


def build_hnsw_index(chunks: List[Chunk], dimension: int) -> FAISSIndex:
    """
    Build FAISS HNSW index from chunks.
    
    Args:
        chunks: List of chunks with embeddings
        dimension: Embedding dimension
    
    Returns:
        FAISSIndex object
    """
    index = FAISSIndex(dimension=dimension, M=32, efConstruction=200)
    index.add_chunks(chunks)
    return index


# =============================================================================
# 6. RETRIEVAL PIPELINE
# =============================================================================

class HybridRetriever:
    """
    Two-stage retrieval:
    1. ANN recall with FAISS HNSW
    2. Reranking with CrossEncoder
    """
    
    def __init__(
        self,
        faiss_index: FAISSIndex,
        embedding_model: SentenceTransformer,
        cross_encoder: CrossEncoder,
        embedding_weight: float = 0.4,
        rerank_weight: float = 0.6
    ):
        """
        Initialize retriever.
        
        Args:
            faiss_index: FAISS index
            embedding_model: SentenceTransformer for query encoding
            cross_encoder: CrossEncoder for reranking
            embedding_weight: Weight for embedding score
            rerank_weight: Weight for rerank score
        """
        self.faiss_index = faiss_index
        self.embedding_model = embedding_model
        self.cross_encoder = cross_encoder
        self.embedding_weight = embedding_weight
        self.rerank_weight = rerank_weight
    
    def recall(self, query: str, top_k: int = 50) -> Tuple[List[str], List[float], List[int]]:
        """
        Stage 1: Recall using ANN vector search.
        
        Args:
            query: Query text
            top_k: Number of candidates to recall
        
        Returns:
            (texts, scores, indices)
        """
        # Encode query
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Search FAISS
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Get texts
        texts = [self.faiss_index.chunk_texts[i] for i in indices if i < len(self.faiss_index.chunk_texts)]
        
        return texts, scores.tolist(), indices.tolist()
    
    def rerank(self, query: str, texts: List[str]) -> List[float]:
        """
        Stage 2: Rerank using CrossEncoder.
        
        Args:
            query: Query text
            texts: Candidate texts
        
        Returns:
            Rerank scores
        """
        # Create query-text pairs
        pairs = [[query, text] for text in texts]
        
        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        
        return scores.tolist()
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Full retrieval pipeline: recall + rerank.
        
        Args:
            query: Query text
            top_k: Number of final results
        
        Returns:
            List of RetrievalResult objects
        """
        # Stage 1: Recall
        texts, emb_scores, indices = self.recall(query, top_k=50)
        
        if not texts:
            return []
        
        # Stage 2: Rerank
        rerank_scores = self.rerank(query, texts)
        
        # Normalize embedding scores to [0, 1] (cosine similarity can be [-1, 1])
        emb_scores_norm = [(score + 1) / 2 for score in emb_scores]  # Map [-1,1] -> [0,1]
        
        # Normalize rerank scores using sigmoid to [0, 1]
        import math
        rerank_scores_norm = [1 / (1 + math.exp(-score)) for score in rerank_scores]
        
        # Combine normalized scores (40% embedding, 60% rerank)
        combined_scores = [
            self.embedding_weight * emb + self.rerank_weight * rerank
            for emb, rerank in zip(emb_scores_norm, rerank_scores_norm)
        ]
        
        # Create results
        results = []
        for i, (text, combined, emb_norm, rerank_norm, idx) in enumerate(zip(
            texts, combined_scores, emb_scores_norm, rerank_scores_norm, indices
        )):
            if idx < len(self.faiss_index.chunk_ids):
                result = RetrievalResult(
                    chunk_id=self.faiss_index.chunk_ids[idx],
                    text=text,
                    combined_score=combined,  # Now guaranteed [0, 1]
                    embedding_score=emb_norm,  # Normalized [0, 1]
                    rerank_score=rerank_norm,  # Normalized [0, 1]
                    metadata=self.faiss_index.metadata[idx]
                )
                results.append(result)
        
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return results[:top_k]


# =============================================================================
# 7. MAIN RAG PIPELINE
# =============================================================================

class RAGPipeline:
    """Complete RAG pipeline: ingest -> index -> retrieve"""
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embedding_model_name: SentenceTransformer model name
            cross_encoder_name: CrossEncoder model name
        """
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        logger.info(f"Loading cross-encoder: {cross_encoder_name}")
        self.cross_encoder = CrossEncoder(cross_encoder_name)
        
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.faiss_index = None
        self.retriever = None
    
    def ingest_document(self, filepath: str) -> List[Chunk]:
        """
        Ingest and chunk a document.
        
        Args:
            filepath: Path to document
        
        Returns:
            List of chunks
        """
        # Load file
        text = load_file(filepath)
        
        # Create chunks
        chunks = create_chunks_from_text(
            text,
            source=Path(filepath).name,
            embedding_model=self.embedding_model
        )
        
        # Embed chunks
        chunks = embed_chunks(chunks, self.embedding_model)
        
        return chunks
    
    def build_index(self, chunks: List[Chunk]):
        """
        Build FAISS index from chunks.
        
        Args:
            chunks: List of chunks with embeddings
        """
        if self.faiss_index is None:
            self.faiss_index = build_hnsw_index(chunks, self.dimension)
        else:
            self.faiss_index.add_chunks(chunks)
        
        # Initialize retriever
        self.retriever = HybridRetriever(
            faiss_index=self.faiss_index,
            embedding_model=self.embedding_model,
            cross_encoder=self.cross_encoder
        )
    
    def query(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Query the RAG system.
        
        Args:
            query: Query text
            top_k: Number of results
        
        Returns:
            List of RetrievalResult objects
        """
        if self.retriever is None:
            raise ValueError("No index built. Call build_index() first.")
        
        return self.retriever.retrieve(query, top_k)
    
    def save_index(self, index_path: str, metadata_path: str):
        """Save FAISS index."""
        if self.faiss_index:
            self.faiss_index.save(index_path, metadata_path)
    
    def load_index(self, index_path: str, metadata_path: str):
        """Load FAISS index."""
        self.faiss_index = FAISSIndex(dimension=self.dimension)
        self.faiss_index.load(index_path, metadata_path)
        
        self.retriever = HybridRetriever(
            faiss_index=self.faiss_index,
            embedding_model=self.embedding_model,
            cross_encoder=self.cross_encoder
        )


# =============================================================================
# 8. EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Initialize pipeline
    rag = RAGPipeline()
    
    # Ingest document
    chunks = rag.ingest_document("sample.pdf")
    
    # Build index
    rag.build_index(chunks)
    
    # Query
    results = rag.query("What is the main topic?", top_k=5)
    
    # Print results
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Score: {result.combined_score:.4f}")
        print(f"Text: {result.text[:200]}...")
