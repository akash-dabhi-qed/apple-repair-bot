import streamlit as st
import pandas as pd
import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_chromadb():
    """Load existing ChromaDB and model"""
    try:
        # Load ChromaDB
        chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Load model
        model = SentenceTransformer('all-mpnet-base-v2')
        
        # Get collections
        mac_collection = chroma_client.get_collection("mac_repairs")
        iphone_collection = chroma_client.get_collection("iphone_repairs")
        
        return chroma_client, model, mac_collection, iphone_collection
    
    except Exception as e:
        st.error(f"Error loading ChromaDB: {e}")
        st.error("Please run the ChromaDB fix script separately if needed.")
        st.stop()

def search_repairs(query, device_type, model, mac_collection, iphone_collection, n_results=5):
    """Search for repair information"""
    
    # Generate query embedding
    query_embedding = model.encode([query])
    
    # Choose collection based on device type
    if device_type.lower() == "mac":
        collection = mac_collection
    else:
        collection = iphone_collection
    
    # Search
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    
    return results

def main():
    st.title("🔧 Apple Repair Assistant")
    st.write("Get help with Mac and iPhone repairs")
    
    # Load ChromaDB
    chroma_client, model, mac_collection, iphone_collection = load_chromadb()
    
    # Success message
    st.success(f"✅ Database loaded: {mac_collection.count()} Mac docs, {iphone_collection.count()} iPhone docs")
    
    # User interface
    col1, col2 = st.columns(2)
    
    with col1:
        device_type = st.selectbox("Device Type", ["Mac", "iPhone"])
    
    with col2:
        n_results = st.slider("Number of results", min_value=1, max_value=10, value=5)
    
    # Search query
    query = st.text_input("What repair issue are you having?", 
                         placeholder="e.g., battery replacement, screen repair, won't turn on")
    
    if query:
        with st.spinner("Searching for solutions..."):
            results = search_repairs(query, device_type, model, mac_collection, iphone_collection, n_results)
        
        st.subheader(f"🔍 Top {len(results['documents'][0])} Results for '{query}'")
        
        for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
            with st.expander(f"Result {i+1} (Relevance: {1-distance:.2f})"):
                st.write(doc)

if __name__ == "__main__":
    main()
