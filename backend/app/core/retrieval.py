import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict

class HybridRetriever:
    def __init__(self):
        # 1. Embedding Model (Dense)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        # 2. Reranker Model (Fine-tuned reranker mentioned in resume)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        self.documents = []
        self.bm25 = None
        self.faiss_index = None
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2

    def ingest_documents(self, docs: List[str]):
        """
        Simulates the ingestion of documentation pages.
        """
        self.documents = docs
        
        # Build Sparse Index (BM25)
        tokenized_corpus = [doc.split(" ") for doc in docs]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Build Dense Index (FAISS)
        embeddings = self.encoder.encode(docs, convert_to_numpy=True)
        # Normalize for Cosine Similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        self.faiss_index.add(embeddings)
        
        print(f"Indexed {len(docs)} documents.")

    def search(self, query: str, top_k: int = 10, alpha: float = 0.5):
        """
        Hybrid Vector Search: Combines BM25 and FAISS scores.
        Alpha controls weight: 1.0 = Pure Vector, 0.0 = Pure Keyword.
        """
        # 1. Sparse Retrieval (BM25)
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 2. Dense Retrieval (FAISS)
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        distances, indices = self.faiss_index.search(query_embedding, len(self.documents))
        
        # Map FAISS distances (cosine sim) to correct indices
        dense_scores = np.zeros(len(self.documents))
        for i, idx in enumerate(indices[0]):
            dense_scores[idx] = distances[0][i]

        # 3. Normalization (Min-Max Scaling) to make scores comparable
        def normalize(scores):
            return (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-9)

        norm_bm25 = normalize(bm25_scores)
        norm_dense = normalize(dense_scores)

        # 4. Hybrid Scoring Formula
        # $$ Score = \alpha \cdot Dense + (1 - \alpha) \cdot Sparse $$
        hybrid_scores = (alpha * norm_dense) + ((1 - alpha) * norm_bm25)
        
        # Get Top K candidates
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        candidates = [self.documents[i] for i in top_indices]
        
        return candidates

    def rerank(self, query: str, candidates: List[str], top_n: int = 3):
        """
        Applies Cross-Encoder to refine results.
        This reduces hallucinations by verifying the connection between query and doc.
        """
        pairs = [[query, doc] for doc in candidates]
        scores = self.reranker.predict(pairs)
        
        # Sort by reranker score
        sorted_indices = np.argsort(scores)[::-1][:top_n]
        final_results = [candidates[i] for i in sorted_indices]
        
        return final_results