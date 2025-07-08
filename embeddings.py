import pandas as pd
import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Load your embedding model
model = SentenceTransformer('all-mpnet-base-v2')

# Initialize Chroma with PERSISTENT storage
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

# Rest of your code remains the same...
mac_collection = chroma_client.create_collection("mac_repairs")
iphone_collection = chroma_client.create_collection("iphone_repairs")

# Load your clean sample data
mac_sample = pd.read_csv('./chunks/mac_chunks_clean.csv')
iphone_sample = pd.read_csv('./chunks/iphone_chunks_clean.csv')

# Generate unique IDs (to avoid duplicates)
mac_sample['unique_id'] = mac_sample['chunk_id'].astype(str) + '_' + mac_sample.index.astype(str)
iphone_sample['unique_id'] = iphone_sample['chunk_id'].astype(str) + '_' + iphone_sample.index.astype(str)

# Generate and store embeddings
print("Generating Mac embeddings...")
mac_embeddings = model.encode(mac_sample['text'].tolist())
mac_collection.add(
    embeddings=mac_embeddings.tolist(),
    documents=mac_sample['text'].tolist(),
    ids=mac_sample['unique_id'].tolist()
)

print("Generating iPhone embeddings...")
iphone_embeddings = model.encode(iphone_sample['text'].tolist())
iphone_collection.add(
    embeddings=iphone_embeddings.tolist(),
    documents=iphone_sample['text'].tolist(),
    ids=iphone_sample['unique_id'].tolist()
)

print("✅ Embeddings stored successfully!")
