#!/usr/bin/env python3
"""
Test script for RAG Pipeline
Run this to test your RAG components without Streamlit
"""

import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from typing import List, Dict

def test_chromadb_connection():
    """Test ChromaDB connection and collections"""
    print("🔍 Testing ChromaDB connection...")
    
    try:
        # Load ChromaDB
        chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get collections
        mac_collection = chroma_client.get_collection("mac_repairs")
        iphone_collection = chroma_client.get_collection("iphone_repairs")
        
        print(f"✅ Mac collection: {mac_collection.count()} documents")
        print(f"✅ iPhone collection: {iphone_collection.count()} documents")
        
        return chroma_client, mac_collection, iphone_collection
    
    except Exception as e:
        print(f"❌ ChromaDB connection failed: {e}")
        return None, None, None

def test_embedding_model():
    """Test embedding model"""
    print("\n🔍 Testing embedding model...")
    
    try:
        model = SentenceTransformer('all-mpnet-base-v2')
        test_text = "battery replacement MacBook Pro"
        embedding = model.encode([test_text])
        
        print(f"✅ Model loaded successfully")
        print(f"✅ Embedding dimension: {embedding.shape[1]}")
        print(f"✅ Test embedding created for: '{test_text}'")
        
        return model
    
    except Exception as e:
        print(f"❌ Embedding model failed: {e}")
        return None

def test_retrieval(model, mac_collection, iphone_collection):
    """Test retrieval functionality"""
    print("\n🔍 Testing retrieval...")
    
    test_queries = [
        ("battery replacement", "mac"),
        ("screen repair", "iphone"),
        ("won't turn on", "mac")
    ]
    
    for query, device_type in test_queries:
        print(f"\n📝 Testing query: '{query}' for {device_type}")
        
        try:
            # Generate embedding
            query_embedding = model.encode([query])
            
            # Choose collection
            collection = mac_collection if device_type == "mac" else iphone_collection
            
            # Search
            results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=3
            )
            
            print(f"✅ Retrieved {len(results['documents'][0])} documents")
            
            # Show top result
            if results['documents'][0]:
                top_doc = results['documents'][0][0]
                similarity = 1 - results['distances'][0][0]
                print(f"   📄 Top result (similarity: {similarity:.3f}): {top_doc[:100]}...")
            
        except Exception as e:
            print(f"❌ Retrieval failed for '{query}': {e}")

def test_groq_connection(api_key):
    """Test Groq API connection"""
    print("\n🔍 Testing Groq connection...")

    if not api_key:
        print("⚠️ No API key provided - skipping Groq test")
        return None

    try:
        client = Groq(api_key=api_key)

        # Test with a simple completion
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "Say 'API test successful'"}],
            max_tokens=10
        )

        print(f"✅ Groq API connected successfully")
        print(f"✅ Test response: {response.choices[0].message.content}")

        return client

    except Exception as e:
        print(f"❌ Groq connection failed: {e}")
        return None

def test_full_rag_pipeline(model, mac_collection, iphone_collection, groq_client):
    print("\n🔍 Testing full RAG pipeline...")

    if not groq_client:
        print("⚠️ No Groq client - skipping full pipeline test")
        return

    test_query = "MacBook Pro battery draining quickly"
    device_type = "mac"

    print(f"📝 Test query: '{test_query}'")

    try:
        # Step 1: Retrieval
        query_embedding = model.encode([test_query])
        collection = mac_collection if device_type.lower() == "mac" else iphone_collection

        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=3
        )

        print(f"✅ Step 1 - Retrieved {len(results['documents'][0])} documents")

        # Step 2: Create context
        context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(results['documents'][0])])

        # Step 3: Create prompt
        prompt = f"""You are an Apple repair expert. Help with this repair issue using the provided documentation.

ISSUE: {test_query}

DOCUMENTATION:
{context}

Provide a brief repair solution:"""

        print(f"✅ Step 2 - Created prompt ({len(prompt)} characters)")

        # Step 4: Get LLM response
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3
        )

        llm_response = response.choices[0].message.content
        print(f"✅ Step 3 - Got LLM response ({len(llm_response)} characters)")
        print(f"📄 Response preview: {llm_response[:200]}...")

        print("\n🎉 Full RAG pipeline test successful!")

    except Exception as e:
        print(f"❌ Full RAG pipeline failed: {e}")

def main():
    print("🧪 RAG Pipeline Test Suite")
    print("=" * 50)
    
    # Test 1: ChromaDB
    chroma_client, mac_collection, iphone_collection = test_chromadb_connection()
    if not chroma_client:
        print("❌ ChromaDB test failed - stopping")
        return
    
    # Test 2: Embedding model
    model = test_embedding_model()
    if not model:
        print("❌ Embedding model test failed - stopping")
        return
    
    # Test 3: Retrieval
    test_retrieval(model, mac_collection, iphone_collection)
    
    # Test 4: OpenAI (optional)
    api_key = input("\n🔑 Enter Groq API key (or press Enter to skip): ").strip()
    groq_client = test_groq_connection(api_key)
    
    # Test 5: Full pipeline (if OpenAI available)
    test_full_rag_pipeline(model, mac_collection, iphone_collection, groq_client)
    
    print("\n✅ Test suite completed!")
    print("\nIf all tests passed, your RAG pipeline is ready!")
    print("Run: streamlit run streamlit_RAG_APP.py")

if __name__ == "__main__":
    main()
