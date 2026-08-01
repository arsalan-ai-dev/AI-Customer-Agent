import os
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

class HybridRetriever:
    def __init__(self, persist_directory="./chroma_db", collection_name="documents"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_collection(collection_name)
        
        all_data = self.collection.get()
        
        if len(all_data['ids']) == 0:
            self.doc_texts = []
            self.bm25 = None
            print("Warning: ChromaDB is empty. Please run ingest.py first.")
            return
            
        self.doc_texts = all_data['documents']
        tokenized_docs = [doc.split(" ") for doc in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.ids = all_data['ids']

    def retrieve(self, query, top_k=5):
        if self.bm25 is None or len(self.doc_texts) == 0:
            return []

        vector_results = self.collection.query(
            query_texts=[query], 
            n_results=20
        )
        
        vector_docs = vector_results['documents'][0]
        vector_ids = vector_results['ids'][0]

        query_tokens = query.split(" ")
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_bm25_indices = np.argsort(bm25_scores)[-20:][::-1]
        bm25_docs = [self.doc_texts[i] for i in top_bm25_indices]
        bm25_ids = [self.ids[i] for i in top_bm25_indices]

        combined_docs = []
        combined_ids = []
        seen = set()

        for i, doc_id in enumerate(vector_ids):
            if doc_id not in seen:
                seen.add(doc_id)
                combined_docs.append(vector_docs[i])
                combined_ids.append(doc_id)
        
        for i, doc_id in enumerate(bm25_ids):
            if doc_id not in seen:
                seen.add(doc_id)
                combined_docs.append(bm25_docs[i])
                combined_ids.append(doc_id)

        if len(combined_docs) > 0:
            pairs = [[query, doc] for doc in combined_docs]
            scores = reranker.predict(pairs)
            
            sorted_indices = np.argsort(scores)[::-1]
            
            final_docs = [combined_docs[i] for i in sorted_indices[:top_k]]
            final_scores = [scores[i] for i in sorted_indices[:top_k]]
            
            print(f"Hybrid Reranking complete! Top score: {max(final_scores):.4f}")
            return final_docs
        else:
            return []

if __name__ == "__main__":
    retriever = HybridRetriever()
    results = retriever.retrieve("What is fair use in copyright?", top_k=3)
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc[:100]}...")