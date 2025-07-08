from langchain_text_splitters import SentenceTransformersTokenTextSplitter
import numpy as np
import pandas as pd
import faiss

class RepairManualRetriever:
    def __init__(self):
        # Load FAISS index
        self.index = faiss.read_index("data/repair_index.faiss")
        
        # Load metadata
        self.metadata = pd.read_csv("data/embeddings_metadata.csv")
        
        # Initialize embedding model
        self.model = SentenceTransformersTokenTextSplitter('all-mpnet-base-v2')
    
    def search(self, query, k=3):
        # Embed the query
        query_embedding = self.model.encode(query)
        
        # Search the index
        distances, indices = self.index.search(
            np.array([query_embedding]), k
        )
        
        # Return top results with metadata
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0:  # FAISS returns -1 for invalid indices
                result = self.metadata.iloc[idx].to_dict()
                result["distance"] = float(distance)
                results.append(result)
        
        return sorted(results, key=lambda x: x["distance"])
