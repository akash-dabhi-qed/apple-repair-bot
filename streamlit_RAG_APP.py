import streamlit as st
import pandas as pd
import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from typing import List, Dict
import json

# Configure page
st.set_page_config(
    page_title="Apple Repair Assistant",
    page_icon="🔧",
    layout="wide"
)

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

def setup_groq_client():
    """Setup Groq client with API key"""
    api_key = st.session_state.get('groq_api_key', '')
    
    if not api_key:
        return None
    
    try:
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error setting up Groq client: {e}")
        return None

def retrieve_relevant_docs(query: str, device_type: str, model, mac_collection, iphone_collection, n_results: int = 5) -> List[Dict]:
    """
    Step 1: Retrieval Logic - Similarity Search
    Retrieve relevant documents based on user query
    """
    
    # Generate query embedding
    query_embedding = model.encode([query])
    
    # Choose collection based on device type
    if device_type.lower() == "mac":
        collection = mac_collection
        device_name = "Mac"
    else:
        collection = iphone_collection
        device_name = "iPhone"
    
    # Perform similarity search
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    
    # Format results
    formatted_results = []
    for i, (doc, distance, doc_id) in enumerate(zip(
        results['documents'][0], 
        results['distances'][0], 
        results['ids'][0]
    )):
        formatted_results.append({
            'content': doc,
            'similarity_score': 1 - distance,  # Convert distance to similarity
            'doc_id': doc_id,
            'rank': i + 1,
            'device_type': device_name
        })
    
    return formatted_results

def create_repair_prompt(query: str, relevant_docs: List[Dict], device_type: str) -> str:
    """
    Step 2: Create Prompt Templates for Repair Context
    Generate a structured prompt for the LLM
    """
    
    # Build context from retrieved documents
    context_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        context_parts.append(f"Document {i} (Relevance: {doc['similarity_score']:.2f}):\n{doc['content']}\n")
    
    context = "\n".join(context_parts)
    
    # Create structured prompt
    prompt = f"""You are an expert Apple repair technician assistant. Help users with {device_type} repair issues based on the provided repair documentation.

REPAIR QUERY: {query}

RELEVANT REPAIR DOCUMENTATION:
{context}

INSTRUCTIONS:
1. Analyze the user's repair query and the provided documentation
2. Provide a clear, step-by-step repair solution if available in the documentation
3. Include important safety warnings and precautions
4. Mention required tools and parts if specified
5. If the documentation doesn't contain enough information, clearly state this
6. Format your response with clear headings and bullet points
7. Be specific about {device_type} models when relevant

RESPONSE FORMAT:
## 🔧 Repair Solution for {device_type}

### 📋 Summary
[Brief summary of the issue and solution]

### ⚠️ Safety Precautions
[Important safety warnings]

### 🛠️ Required Tools & Parts
[List tools and parts needed]

### 📝 Step-by-Step Instructions
[Detailed repair steps]

### 💡 Additional Tips
[Helpful tips and troubleshooting]

### ⚡ When to Seek Professional Help
[Situations requiring professional repair]

Please provide your repair guidance:"""

    return prompt

def get_llm_response(prompt: str, client, model_name: str = "llama-3.1-8b-instant") -> str:
    """
    Step 3: LLM Integration
    Get response from Groq API
    """
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert Apple repair technician with years of experience fixing Mac and iPhone devices. You provide clear, safe, and accurate repair guidance."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.3  # Lower temperature for more consistent technical responses
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error getting LLM response: {str(e)}"

def rag_pipeline(query: str, device_type: str, model, mac_collection, iphone_collection, groq_client, n_results: int = 5) -> Dict:
    """
    Step 4: Complete RAG Pipeline
    Combines retrieval, prompt creation, and LLM response
    """
    
    # Step 1: Retrieve relevant documents
    relevant_docs = retrieve_relevant_docs(query, device_type, model, mac_collection, iphone_collection, n_results)
    
    # Step 2: Create prompt with context
    prompt = create_repair_prompt(query, relevant_docs, device_type)
    
    # Step 3: Get LLM response
    llm_response = get_llm_response(prompt, groq_client)
    
    # Return complete pipeline result
    return {
        'query': query,
        'device_type': device_type,
        'retrieved_docs': relevant_docs,
        'prompt': prompt,
        'response': llm_response,
        'num_docs_used': len(relevant_docs)
    }

def main():
    st.title("🔧 Apple Repair Assistant with RAG")
    st.markdown("Get expert repair guidance powered by AI and comprehensive repair documentation")
    
    # Load ChromaDB
    chroma_client, model, mac_collection, iphone_collection = load_chromadb()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Groq API Key
        st.subheader("🔑 Groq API Key")
        api_key = st.text_input(
            "Enter your Groq API key:",
            type="password",
            help="Get your API key from https://console.groq.com/keys"
        )
        
        if api_key:
            st.session_state['groq_api_key'] = api_key
            st.success("✅ API key configured")
        else:
            st.warning("⚠️ Please enter your Groq API key to use the AI assistant")
        
        # Model selection
        st.subheader("🤖 Model Settings")
        model_name = st.selectbox(
            "Groq Model:",
            [
                "mixtral-8x7b-32768",     # Fast + high quality
                "llama3-70b-8192",        # Very strong general model
                "gemma-7b-it",            # Lightweight option
                "llama2-70b-4096"         # Legacy LLaMA model
            ],
            index=0,
            help="Groq models are hosted for free and optimized for speed"
        )

        # Retrieval settings
        st.subheader("🔍 Retrieval Settings")
        n_results = st.slider(
            "Documents to retrieve:",
            min_value=1,
            max_value=10,
            value=5,
            help="More documents provide more context but may exceed token limits"
        )
        
        # Database info
        st.subheader("📊 Database Info")
        st.info(f"Mac documents: {mac_collection.count()}")
        st.info(f"iPhone documents: {iphone_collection.count()}")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask Your Repair Question")
        
        # Device selection
        device_type = st.selectbox(
            "Select Device Type:",
            ["Mac", "iPhone"],
            help="Choose the device you need help with"
        )
        
        # Query input
        query = st.text_area(
            "Describe your repair issue:",
            placeholder="e.g., My MacBook Pro won't turn on, battery drains quickly, screen is cracked, keyboard keys not working...",
            height=100
        )
        
        # Search button
        search_button = st.button("🔍 Get Repair Guidance", type="primary")
    
    with col2:
        st.subheader("📝 Quick Examples")
        st.markdown("""
        **Mac Issues:**
        - Battery replacement steps
        - Screen flickering problems
        - Keyboard not responding
        - Won't boot up
        
        **iPhone Issues:**
        - Cracked screen repair
        - Battery draining fast
        - Camera not working
        - Water damage recovery
        """)
    
    # Process query
    if search_button and query:
        if not st.session_state.get('groq_api_key'):
            st.error("❌ Please enter your Groq API key in the sidebar")
            return
        
        # Setup Groq client
        groq_client = setup_groq_client()
        if not groq_client:
            st.error("❌ Failed to setup Groq client")
            return
        
        # Show processing
        with st.spinner(f"🔍 Searching repair documentation and generating guidance..."):
            
            # Run RAG pipeline
            result = rag_pipeline(
                query=query,
                device_type=device_type,
                model=model,
                mac_collection=mac_collection,
                iphone_collection=iphone_collection,
                groq_client=groq_client,
                n_results=n_results
            )
        
        # Display results
        st.subheader("🤖 AI Repair Guidance")
        st.markdown(result['response'])
        
        # Show retrieved documents in expander
        with st.expander(f"📚 Retrieved Documentation ({result['num_docs_used']} documents)", expanded=False):
            for doc in result['retrieved_docs']:
                st.markdown(f"**Document {doc['rank']} - Relevance: {doc['similarity_score']:.2f}**")
                st.markdown(f"```\n{doc['content'][:500]}{'...' if len(doc['content']) > 500 else ''}\n```")
                st.markdown("---")
        
        # Debug info in expander
        with st.expander("🔧 Debug Information", expanded=False):
            st.json({
                'query': result['query'],
                'device_type': result['device_type'],
                'num_docs_retrieved': result['num_docs_used'],
                'model_used': model_name,
                'prompt_length': len(result['prompt'])
            })
    
    elif search_button:
        st.warning("⚠️ Please enter a repair question")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔧 Apple Repair Assistant - Powered by RAG (Retrieval-Augmented Generation)</p>
        <p><strong>Disclaimer:</strong> This is for educational purposes. Always consult professionals for complex repairs.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
