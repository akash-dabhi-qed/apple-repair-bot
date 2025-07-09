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
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import time
from groq import Groq, APITimeoutError, APIStatusError

# Define the maximum number of messages to keep in history for the LLM context.
# This helps manage token limits. Adjust as needed.
MAX_LLM_HISTORY_LENGTH = 10 # Keep 10 messages (5 user, 5 assistant) for LLM context

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="🔧 AI Repair Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .result-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        box_shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .device-badge-mac {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .device-badge-iphone {
        background: #f3e5f5;
        color: #7b1fa2;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .relevance-high { color: #28a745; font-weight: bold; }
    .relevance-medium { color: #ffc107; font-weight: bold; }
    .relevance-low { color: #dc3545; font-weight: bold; }
    
    .debug-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    .ai-response {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        box_shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_chromadb():
    """Load existing ChromaDB and model with enhanced error handling"""
    try:
        # Check if ChromaDB path exists
        db_path = "./chroma_db"
        if not os.path.exists(db_path):
            return None, None, None, None, f"ChromaDB path not found: {db_path}"
        
        # Load ChromaDB
        chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Load model
        model = SentenceTransformer('all-mpnet-base-v2')
        
        # Try to get collections - handle if they don't exist
        try:
            mac_collection = chroma_client.get_collection("mac_repairs")
        except Exception as e:
            mac_collection = None
            st.warning(f"Mac collection not found: {e}")
        
        try:
            iphone_collection = chroma_client.get_collection("iphone_repairs")
        except Exception as e:
            iphone_collection = None
            st.warning(f"iPhone collection not found: {e}")
        
        return chroma_client, model, mac_collection, iphone_collection, None
    
    except Exception as e:
        return None, None, None, None, str(e)

@st.cache_data
def get_collection_stats():
    """Get collection statistics (cached)"""
    try:
        _, _, mac_collection, iphone_collection, error = load_chromadb()
        
        if error:
            return {"error": error}
        
        mac_count = mac_collection.count() if mac_collection else 0
        iphone_count = iphone_collection.count() if iphone_collection else 0
        
        return {
            "mac_count": mac_count,
            "iphone_count": iphone_count,
            "total_count": mac_count + iphone_count,
            "error": None
        }
    except Exception as e:
        return {"error": str(e)}

def debug_collections():
    """Debug function to check collection status"""
    try:
        chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # List all collections
        collections = chroma_client.list_collections()
        collection_names = [col.name for col in collections]
        
        debug_info = {
            "db_path_exists": os.path.exists("./chroma_db"),
            "collections_found": collection_names,
            "expected_collections": ["mac_repairs", "iphone_repairs"]
        }
        
        # Get individual collection info
        for col_name in ["mac_repairs", "iphone_repairs"]:
            try:
                col = chroma_client.get_collection(col_name)
                debug_info[f"{col_name}_count"] = col.count()
                
                # Test a sample query
                if col.count() > 0:
                    sample_query = col.query(
                        query_texts=["battery replacement"],
                        n_results=1
                    )
                    debug_info[f"{col_name}_sample_query_success"] = True
                    debug_info[f"{col_name}_sample_result"] = sample_query['documents'][0][0][:100] + "..." if sample_query['documents'][0] else "No results"
                else:
                    debug_info[f"{col_name}_sample_query_success"] = False
                    debug_info[f"{col_name}_sample_result"] = "Collection is empty"
                    
            except Exception as e:
                debug_info[f"{col_name}_error"] = str(e)
        
        return debug_info
    except Exception as e:
        return {"error": str(e)}

@st.cache_resource
def setup_groq_client(api_key: str):
    """Setup Groq client with API key"""
    api_key = st.session_state.get('groq_api_key', '')
    
    if api_key:
        try:
            client = Groq(api_key=api_key)
            # Optional: Add a light API call here to immediately test the key's validity
            # client.models.list()
            return client
        except Exception as e:
            st.error(f"Error initializing Groq client with the provided key: {e}. Please check your API key.")
            return None
    return None

def retrieve_relevant_docs(query: str, device_type: str, model, mac_collection, iphone_collection, n_results: int = 5, min_relevance: float = 0.3, show_debug: bool = False) -> List[Dict]:
    """
    Enhanced retrieval with debug info and relevance filtering
    """
    
    if not query.strip():
        return []
    
    # Debug info
    if show_debug:
        st.write(f"🔍 **Debug Info:**")
        st.write(f"- Query: '{query}'")
        st.write(f"- Device type: {device_type}")
        st.write(f"- Mac collection available: {mac_collection is not None}")
        st.write(f"- iPhone collection available: {iphone_collection is not None}")
        
        if mac_collection:
            st.write(f"- Mac collection count: {mac_collection.count()}")
        if iphone_collection:
            st.write(f"- iPhone collection count: {iphone_collection.count()}")
    
    # Generate query embedding
    try:
        query_embedding = model.encode([query])
        if show_debug:
            st.write(f"- Query embedding generated successfully: shape {query_embedding.shape}")
    except Exception as e:
        st.error(f"Error generating query embedding: {e}")
        return []
    
    results = []
    
    # Search based on device type
    if device_type.lower() == "mac" and mac_collection:
        try:
            if show_debug:
                st.write("🔍 Searching Mac collection...")
            mac_results = mac_collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results
            )
            
            if show_debug:
                st.write(f"- Mac query returned {len(mac_results['documents'][0])} results")
            
            for i, (doc, distance, doc_id) in enumerate(zip(
                mac_results['documents'][0], 
                mac_results['distances'][0],
                mac_results['ids'][0]
            )):
                relevance_score = 1 - distance
                if show_debug:
                    st.write(f"  - Result {i+1}: relevance={relevance_score:.3f}, distance={distance:.3f}")
                
                if relevance_score >= min_relevance:
                    results.append({
                        'content': doc,
                        'similarity_score': relevance_score,
                        'doc_id': doc_id,
                        'rank': i + 1,
                        'device_type': 'Mac',
                        'distance': distance
                    })
        except Exception as e:
            st.error(f"Error searching Mac collection: {e}")
    
    elif device_type.lower() == "iphone" and iphone_collection:
        try:
            if show_debug:
                st.write("🔍 Searching iPhone collection...")
            iphone_results = iphone_collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results
            )
            
            if show_debug:
                st.write(f"- iPhone query returned {len(iphone_results['documents'][0])} results")
            
            for i, (doc, distance, doc_id) in enumerate(zip(
                iphone_results['documents'][0], 
                iphone_results['distances'][0],
                iphone_results['ids'][0]
            )):
                relevance_score = 1 - distance
                if show_debug:
                    st.write(f"  - Result {i+1}: relevance={relevance_score:.3f}, distance={distance:.3f}")
                
                if relevance_score >= min_relevance:
                    results.append({
                        'content': doc,
                        'similarity_score': relevance_score,
                        'doc_id': doc_id,
                        'rank': i + 1,
                        'device_type': 'iPhone',
                        'distance': distance
                    })
        except Exception as e:
            st.error(f"Error searching iPhone collection: {e}")
    
    elif device_type.lower() == "both":
        # Search both collections and combine results
        all_results = []
        if mac_collection:
            try:
                if show_debug:
                    st.write("🔍 Searching Mac collection (for 'Both')...")
                mac_results = mac_collection.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=n_results
                )
                for i, (doc, distance, doc_id) in enumerate(zip(
                    mac_results['documents'][0], 
                    mac_results['distances'][0],
                    mac_results['ids'][0]
                )):
                    relevance_score = 1 - distance
                    if relevance_score >= min_relevance:
                        all_results.append({
                            'content': doc,
                            'similarity_score': relevance_score,
                            'doc_id': doc_id,
                            'rank': i + 1,
                            'device_type': 'Mac',
                            'distance': distance
                        })
            except Exception as e:
                st.error(f"Error searching Mac collection (for 'Both'): {e}")

        if iphone_collection:
            try:
                if show_debug:
                    st.write("🔍 Searching iPhone collection (for 'Both')...")
                iphone_results = iphone_collection.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=n_results
                )
                for i, (doc, distance, doc_id) in enumerate(zip(
                    iphone_results['documents'][0], 
                    iphone_results['distances'][0],
                    iphone_results['ids'][0]
                )):
                    relevance_score = 1 - distance
                    if relevance_score >= min_relevance:
                        all_results.append({
                            'content': doc,
                            'similarity_score': relevance_score,
                            'doc_id': doc_id,
                            'rank': i + 1,
                            'device_type': 'iPhone',
                            'distance': distance
                        })
            except Exception as e:
                st.error(f"Error searching iPhone collection (for 'Both'): {e}")
        
        # Sort combined results by relevance score (descending) and take top N
        results = sorted(all_results, key=lambda x: x['similarity_score'], reverse=True)[:n_results]

    if show_debug:
        st.write(f"🎯 **Final results:** {len(results)} results above threshold {min_relevance}")
    
    return results

def create_repair_prompt(query: str, relevant_docs: List[Dict], device_type: str) -> str:
    """Create structured prompt for the LLM"""

    # Build context from retrieved documents
    context_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        context_parts.append(f"Document {i} (Relevance: {doc['similarity_score']:.2f}):\n{doc['content']}\n")

    context = "\n".join(context_parts)

    # Create structured prompt
    prompt = f"""You are an expert Apple repair technician assistant. Help users with {device_type} repair issues based on the provided repair documentation.
    Your goal is to provide a structured repair solution in JSON format.

    REPAIR QUERY: {query}

    RELEVANT REPAIR DOCUMENTATION:
    {context}

    INSTRUCTIONS:
    1. Analyze the user's repair query and the provided documentation.
    2. Provide a clear, step-by-step repair solution if available in the documentation.
    3. Include important safety warnings and precautions.
    4. Mention required tools and parts if specified.
    5. If the documentation doesn't contain enough information, clearly state this.
    6. **IMPORTANT:** Your response MUST be a valid JSON object with the following keys. If a section is not applicable, you can leave its value as an empty string or null.
    7. **CRITICAL:** Ensure all double quotes WITHIN string values are escaped with a backslash (e.g., "14\\\"" for "14\""). Do NOT use single quotes.

    RESPONSE JSON FORMAT:
    ```json
    {{
        "title": "Repair Solution for {device_type} - [Summary of issue]",
        "summary": "[Brief summary of the issue and solution]",
        "safety_precautions": "[Important safety warnings, e.g., 'Disconnect power before starting.']",
        "required_tools_parts": "[List tools and parts needed, e.g., 'Phillips head screwdriver, new battery.']",
        "step_by_step_instructions": [
            "Step 1: [Detailed first step]",
            "Step 2: [Detailed second step]",
            // ... more steps
        ],
        "additional_tips": "[Helpful tips and troubleshooting advice]",
        "professional_help_situations": "[Situations requiring professional repair, e.g., 'If you encounter liquid damage beyond your skill level.']",
        "notes": "[Any other important notes or disclaimers.]"
    }}"""

    return prompt

def get_llm_response(
    chat_history: List[Dict],
    client: Groq,
    model_name: str,
    device_type: str = "Apple device"
): # Removed -> str type hint as it now returns an iterator
    """Get response from Groq API using chat history for context, with streaming."""
    try:
        if device_type.lower() == "mac":
            specific_device_phrase = "Mac repair technician specializing in Mac computers."
            focus_device_phrase = "Mac computer repair guidance."
        elif device_type.lower() == "iphone":
            specific_device_phrase = "iPhone repair technician specializing in iPhone mobile devices."
            focus_device_phrase = "iPhone mobile device repair guidance."
        else:
            # Fallback for "both" or any other unexpected device_type
            specific_device_phrase = "Apple repair technician with years of experience fixing Mac and iPhone devices."
            focus_device_phrase = "Apple device repair guidance."

        system_message = {
        "role": "system",
        "content": (
            f"You are an expert {specific_device_phrase} You provide clear, safe, and accurate repair guidance. "
            "Always prioritize user safety and instruct users to seek professional help for complex or dangerous repairs. "
            f"Focus primarily on {focus_device_phrase}, but you can also provide general advice, safety tips, and assess repair difficulty based on common knowledge or inferred complexity from the provided documentation, tailored to various skill levels. "
            "\n\n**IMPORTANT:** If the user's query is inappropriate, off-topic, or asks for assistance outside of Apple device repair..., you MUST politely decline the request and state that you can only assist with Apple device repair."
        )
    }

        # Define your few-shot examples here based on the provided outputs
        few_shot_examples = [
            # Example 1: MacBook Pro Fan Replacement
            {
                "role": "user",
                "content": create_repair_prompt(
                    query="My MacBook Pro fans are making a lot of noise and I think they need replacement. Can you guide me?",
                    relevant_docs=[
                        {"content": "MacBook Pro 16\" 2023 Fans Replacement: Details on replacing fans for this model, including screws and steps.", "similarity_score": 0.9, "device_type": "Mac", "doc_id": "mac_fan_001", "rank": 1, "distance": 0.1},
                        {"content": "MacBook Pro 16\" Late 2023 Fans Replacement: Specifics for the late 2023 model's fan replacement.", "similarity_score": 0.88, "device_type": "Mac", "doc_id": "mac_fan_002", "rank": 2, "distance": 0.12},
                        {"content": "Mac Pro Late 2013 Fan Replacement: Information regarding fan replacement for Mac Pro.", "similarity_score": 0.7, "device_type": "Mac", "doc_id": "mac_fan_003", "rank": 3, "distance": 0.3},
                    ],
                    device_type="Mac"
                )
            },
            {
                "role": "assistant",
                "content": """```json
{
    "title": "Repair Solution for Both - MacBook Pro Fan Replacement Guide",
    "summary": "This guide will walk you through the process of replacing the fans in your MacBook Pro. Please note that the exact steps may vary depending on the model of your MacBook Pro.",
    "safety_precautions": "Disconnect power before starting. If your battery is swollen, take appropriate precautions to avoid injury.",
    "required_tools_parts": "iFixit Precision Bit Driver, P5 Pentalobe Bit, 5IP Torx Plus Bit, 3IP Torx Plus Bit, T6 Torx Bit, 4 mm Hex Nut Driver Bit, Suction Handle, iFixit Opening Picks (Set of 6), ESD Safe Blunt Nose Tweezers, Spudger, MacBook Pro 16\\" (A2485, A2780, A2991) Left Fan, MacBook Pro 16\\" (A2485, A2991) Right Fan",
    "step_by_step_instructions": [
        "Step 1: Remove the following screws: Three 14.4 mm Phillips #00 screws, Three 3.5 mm Phillips #00 screws, Four 3.5 mm shouldered Phillips #00 screws.",
        "Step 2: Use your fingers to pry the lower case away from the body of the MacBook near the vent. Remove the lower case.",
        "Step 3: Disconnect the battery connector from the logic board.",
        "Step 4: Remove the fan from the MacBook Pro.",
        "Step 5: Clean the area where the new fan will be installed.",
        "Step 6: Apply replacement fan adhesive to the new fan.",
        "Step 7: Install the new fan and reconnect the fan connector to the logic board.",
        "Step 8: Reassemble the MacBook Pro in the reverse order of the steps above."
    ],
    "additional_tips": "Make sure to handle the fan with care to avoid damage. If you encounter any issues during the repair, consider seeking professional help.",
    "professional_help_situations": "If you are not comfortable with this repair or if you encounter any issues during the process, consider seeking professional help from an authorized Apple service provider.",
    "notes": "Please note that the exact steps may vary depending on the model of your MacBook Pro. It's also important to ensure that you have the correct replacement fan and adhesive for your specific model.",
    "source_documents_used": [
        "Doc 1 (Mac - 55.9%): **MacBook Pro 16\\" 2023 Fans Replacement**",
        "Doc 2 (Mac - 54.3%): **MacBook Pro 16\\" Late 2023 Fans Replacement**",
        "Doc 3 (Mac - 54.0%): **Mac Pro Late 2013 Fan Replacement**",
        "Doc 4 (Mac - 51.5%): **Macbook Pro 14\\" Late 2023 (M3 Pro and M3 Max) Fans Replacement**",
        "Doc 5 (Mac - 51.4%): **MacBook Pro 16\\" 2021 Fan Assembly Replacement**"
    ]
}
```"""
            },
            # Example 2: iPhone 13 Speaker Replacement
            {
                "role": "user",
                "content": create_repair_prompt(
                    query="My iPhone 13 speaker is not working. How can I replace it?",
                    relevant_docs=[
                        {"content": "iPhone 13 Pro Max Loudspeaker Replacement: Guide for replacing loudspeaker on iPhone 13 Pro Max.", "similarity_score": 0.95, "device_type": "iPhone", "doc_id": "iphone_speaker_001", "rank": 1, "distance": 0.05},
                        {"content": "iPhone 13 mini Bottom Speaker Replacement: Instructions for replacing the bottom speaker on iPhone 13 mini.", "similarity_score": 0.92, "device_type": "iPhone", "doc_id": "iphone_speaker_002", "rank": 2, "distance": 0.08},
                        {"content": "iPhone 12 Pro Max Loudspeaker Replacement: Relevant steps for iPhone 12 Pro Max loudspeaker.", "similarity_score": 0.8, "device_type": "iPhone", "doc_id": "iphone_speaker_003", "rank": 3, "distance": 0.2},
                    ],
                    device_type="iPhone"
                )
            },
            {
                "role": "assistant",
                "content": """```json
{
    "title": "Repair Solution for Both - iPhone 13 Speaker Replacement",
    "summary": "This guide will walk you through the process of replacing the speaker in your iPhone 13.",
    "safety_precautions": "Disconnect power before starting. Before you begin, discharge your iPhone battery below 25%. A charged lithium-ion battery can catch fire and/or explode if accidentally punctured.",
    "required_tools_parts": "P2 Pentalobe Screwdriver iPhone, Clampy - Anti-Clamp, Hair Dryer, Heat Gun, iFixit Opening Picks (Set of 6), Suction Handle, Tri-point Y000 Screwdriver, Spudger, Tweezers, iPhone 13 Loudspeaker",
    "step_by_step_instructions": [
        "Step 1: Remove the two 6.8 mm-long pentalobe P2 screws at the bottom edge of the iPhone.",
        "Step 2: Use a pair of tweezers to remove the speaker.",
        "Step 3: Compare your new replacement part to the original part—you may need to transfer remaining components or remove adhesive backings from the new part before installing.",
        "Step 4: Apply new adhesive where necessary after cleaning the relevant areas with isopropyl alcohol (>90%)."
    ],
    "additional_tips": "Take your e-waste to an R2 or e-Stewards certified recycler. Repair didn’t go as planned? Try some basic troubleshooting, or ask our iPhone 13 mini Answers community for help.",
    "professional_help_situations": "If you encounter any issues during the process, consider seeking professional help from an authorized Apple service provider.",
    "notes": "Please note that the exact steps may vary depending on the model of your iPhone 13. It's also important to ensure that you have the correct replacement speaker and adhesive for your specific model.",
    "source_documents_used": [
        "Doc 1 (iPhone - 61.9%): **iPhone 13 Pro Max Loudspeaker Replacement**",
        "Doc 2 (iPhone - 55.1%): **iPhone 13 mini Bottom Speaker Replacement**",
        "Doc 3 (iPhone - 52.4%): **iPhone 12 Pro Max Loudspeaker Replacement**",
        "Doc 4 (iPhone - 49.5%): **Unscrew the speaker**",
        "Doc 5 (iPhone - 49.3%): **Step 2**"
    ]
}
```"""
            }
        ]
        
        # Combine system message, few-shot examples, and the current chat history for the LLM.
        # The 'chat_history' should already include the RAG prompt as the last user message from rag_pipeline.
        # Ensure the actual RAG prompt is always the last 'user' message to the LLM.
        messages_for_llm = [system_message] + \
                           few_shot_examples + \
                           list(chat_history)

        stream = client.chat.completions.create(
            messages=messages_for_llm,
            model=model_name,
            temperature=st.session_state.get('temperature', 0.3),
            max_tokens=st.session_state.get('max_tokens', 1500),
            top_p=st.session_state.get('top_p', 1.0),
            stream=True,
            stop=None,
            timeout=st.session_state.get('llm_timeout', 60.0),
        )
        return stream
    
    except APITimeoutError:
        # Handle LLM Timeout
        st.error("❗ The AI took too long to respond. Please try again or adjust timeout settings in the sidebar.")
        return (chunk for chunk in []) # Return empty generator to stop further processing in caller
    except APIStatusError as e:
        # Handle API Status Errors, including Rate Limits (HTTP 429)
        if e.status_code == 429:
            st.error("🚦 Rate limit hit! Too many requests. Please wait a moment before trying again, or consider upgrading your Groq plan if this happens frequently.")
        else:
            # Handle other HTTP errors (e.g., 400 Bad Request, 500 Internal Server Error)
            error_detail = e.response.json().get('detail', 'Unknown API error') if e.response else 'No detailed response'
            st.error(f"⚠️ Groq API error ({e.status_code}): {error_detail}")
        return (chunk for chunk in []) # Return empty generator
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return (chunk for chunk in [])
    
def detect_device_type(query: str) -> str:
    """Auto-detect device type from query"""
    query_lower = query.lower()
    
    mac_keywords = ['mac', 'imac', 'macbook', 'mac mini', 'mac pro', 'macbook air', 'macbook pro']
    iphone_keywords = ['iphone', 'ios', 'apple phone']
    
    mac_score = sum(1 for keyword in mac_keywords if keyword in query_lower)
    iphone_score = sum(1 for keyword in iphone_keywords if keyword in query_lower)
    
    if mac_score > iphone_score:
        return "mac"
    elif iphone_score > mac_score:
        return "iphone"
    else:
        return "both"

def get_relevance_class(score: float) -> str:
    """Get CSS class for relevance score"""
    if score >= 0.7:
        return "relevance-high"
    elif score >= 0.5:
        return "relevance-medium"
    else:
        return "relevance-low"

def render_result_card(result: Dict, index: int):
    """Render a single result card"""
    device_class = "device-badge-mac" if result['device_type'] == 'Mac' else "device-badge-iphone"
    relevance_class = get_relevance_class(result['similarity_score'])
    
    st.markdown(f"""
    <div class="result-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span class="{device_class}">{result['device_type']} Repair</span>
            <span class="{relevance_class}">{result['similarity_score']:.1%} match</span>
        </div>
        <p style="line-height: 1.6; color: #444; margin: 0;">{result['content'][:300]}{'...' if len(result['content']) > 300 else ''}</p>
    </div>
    """, unsafe_allow_html=True)

def process_user_query(
    query_text: str,
    device_type_setting: str,
    auto_detect_setting: bool,
    model_embedding,
    mac_collection_db,
    iphone_collection_db,
    groq_client_obj: Groq,
    n_results_setting: int,
    min_relevance_setting: float,
    model_name_setting: str,
    show_debug_setting: bool
):
    # Add user's query to session state messages for history
    st.session_state.messages.append({"role": "user", "content": query_text})

    # Limit chat history length for LLM context
    # Note: st.session_state.messages already includes the *current* user query
    if len(st.session_state.messages) > MAX_LLM_HISTORY_LENGTH:
        st.session_state.messages = st.session_state.messages[-MAX_LLM_HISTORY_LENGTH:]

    final_device_type = device_type_setting
    if auto_detect_setting:
        detected_type = detect_device_type(query_text)
        if detected_type != "both":
            final_device_type = detected_type.title()

    # --- MODIFIED BLOCK START ---
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_content = ""
        parsing_error_message = ""
        parsed_data = None
        parsing_successful = False
        relevant_docs = [] # Initialize here, will be populated if RAG is used
        generated_rag_prompt = None # Initialize for debug info

        with st.spinner(f"🔍 Processing your request..."):
            # Heuristic to detect if the user's query is a follow-up question
            is_follow_up_question = any(
                keyword in query_text.lower() for keyword in ["how hard", "difficulty", "novice", "expert", "tools", "safety", "dangerous", "what if", "how long", "where can i find", "tips", "troubleshoot", "about this", "explain", "more detail"]
            ) or (
                "above" in query_text.lower() and
                len(st.session_state.messages) > 1 and
                # Check if the immediately preceding assistant message was a repair guide
                "repair guide" in st.session_state.messages[-2]['content'].lower() # Check the content of the *previous assistant message*
            )

            llm_response_generator = None # Initialize generator

            if is_follow_up_question:
                # For follow-up questions, send the existing chat history directly.
                # The user's current query is already appended to st.session_state.messages.
                # We also need to include the system message.
                temp_chat_history_for_llm = list(st.session_state.messages)
                
                # Update placeholder to indicate follow-up processing
                message_placeholder.markdown("Understanding your follow-up question... 💬")

                llm_response_generator = get_llm_response(
                    chat_history=temp_chat_history_for_llm,
                    client=groq_client_obj,
                    device_type=final_device_type, # Still pass device_type for context
                    model_name=model_name_setting
                )
                # No new RAG docs for follow-up questions in this branch
                relevant_docs = []

            else:
                # For new repair requests, proceed with RAG
                message_placeholder.markdown("Searching repair documentation and generating guidance... 🔍")
                relevant_docs = retrieve_relevant_docs(
                    query_text, final_device_type, model_embedding,
                    mac_collection_db, iphone_collection_db,
                    n_results_setting, min_relevance_setting, show_debug_setting
                )

                if not relevant_docs:
                    full_response_content = f"No relevant documentation found for your {final_device_type} repair query. Please try different keywords or lower the relevance threshold."
                    # No LLM call if no relevant docs found for a *new* query
                    llm_response_generator = None
                else:
                    # Create RAG-enhanced prompt
                    generated_rag_prompt = create_repair_prompt(query_text, relevant_docs, final_device_type)
                    
                    # For RAG queries, send the history excluding the *current* user query,
                    # then append the RAG prompt as the last user message.
                    # This ensures the LLM sees the original conversation + the detailed RAG prompt.
                    temp_chat_history_for_llm = list(st.session_state.messages[:-1]) + [{"role": "user", "content": generated_rag_prompt}]

                    llm_response_generator = get_llm_response(
                        chat_history=temp_chat_history_for_llm,
                        client=groq_client_obj,
                        device_type=final_device_type,
                        model_name=model_name_setting
                    )

            if llm_response_generator: # Only proceed with streaming if a generator was created
                # Iterate over streamed chunks and display
                for chunk in llm_response_generator:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_response_content += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response_content + "▌") # Add a blinking cursor effect
                message_placeholder.markdown(full_response_content) # Remove cursor after completion
            else:
                # If no LLM response was generated (e.g., no relevant docs), display the pre-set content
                message_placeholder.markdown(full_response_content)

        # --- NEW PARSING LOGIC START ---
        final_display_content = "" # Ensure it's initialized before the try-block
        
        try:
            # First, try to extract JSON from a markdown block if present
            json_start = full_response_content.find('```json')
            json_end = full_response_content.rfind('```')
            
            json_string_to_parse = full_response_content.strip() # Default to parsing whole response

            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_string_to_parse = full_response_content[json_start + len('```json'):json_end].strip()
            
            parsed_data = json.loads(json_string_to_parse)
            parsing_successful = True

            # Check if it's a structured repair solution (from previous few-shot examples)
            if isinstance(parsed_data, dict) and "repair_solution" in parsed_data:
                # Handle the structured repair solution format
                st.subheader(f"🛠️ {parsed_data.get('device', 'Device')} Repair Guide: {parsed_data.get('issue', 'Issue')}")

                st.warning("⚠️ **Safety First!** Always unplug devices and take necessary precautions before starting any repair. If you are unsure, consult a professional.")

                if "overview" in parsed_data["repair_solution"] and parsed_data["repair_solution"]["overview"]:
                    st.markdown(f"**Overview:** {parsed_data['repair_solution']['overview']}")

                if "required_tools" in parsed_data["repair_solution"] and parsed_data["repair_solution"]["required_tools"]:
                    st.markdown("**Required Tools:**")
                    st.markdown(", ".join([f"• {tool}" for tool in parsed_data["repair_solution"]["required_tools"]]))

                if "difficulty" in parsed_data["repair_solution"] and parsed_data["repair_solution"]["difficulty"]:
                    st.markdown(f"**Difficulty:** {parsed_data['repair_solution']['difficulty']}")

                if "step_by_step_instructions" in parsed_data["repair_solution"] and parsed_data["repair_solution"]["step_by_step_instructions"]:
                    st.markdown("**Repair Steps:**")
                    for i, step in enumerate(parsed_data["repair_solution"]["step_by_step_instructions"]):
                        st.markdown(f"**Step {i+1}: {step.get('title', '')}**")
                        st.write(step.get('description', ''))

                if "additional_tips" in parsed_data["repair_solution"] and parsed_data["repair_solution"]["additional_tips"]:
                    st.markdown("**Additional Tips:**")
                    for tip in parsed_data["repair_solution"]["additional_tips"]:
                        st.markdown(f"- {tip}")
                
                # Append relevant docs here if it's a RAG response (meaning relevant_docs was populated)
                if relevant_docs: 
                    st.markdown("\n\n**📚 Source Documents Used:**\n")
                    for i, doc in enumerate(relevant_docs):
                        doc_content = doc['content']
                        doc_name = "Untitled Document" # Default name
                        lines = doc_content.strip().split('\n')
                        if lines:
                            first_line = lines[0].strip()
                            if first_line.startswith('#'):
                                doc_name = first_line.lstrip('# ').strip()
                            elif first_line:
                                doc_name = first_line
                            else: # If first line is empty, try second
                                if len(lines) > 1:
                                    second_line = lines[1].strip()
                                    if second_line.startswith('#'):
                                        doc_name = second_line.lstrip('# ').strip()
                                    elif second_line:
                                        doc_name = second_line
                        if len(doc_name) > 100:
                            doc_name = doc_name[:97] + "..."
                        st.markdown(f"- Doc {i+1} ({doc['device_type']} - {doc['similarity_score']:.1%}): **{doc_name}**")
                
                final_display_content = f"🛠️ Repair guide for {parsed_data.get('device', 'device')} issue: {parsed_data.get('issue', 'N/A')}. Please see details above." # Summary for history
                
            elif isinstance(parsed_data, dict):
                # If it's a dictionary but not the specific repair_solution format,
                # display it as a general JSON response (e.g., debug info from LLM)
                st.subheader("💡 AI Response (JSON):")
                st.json(parsed_data)
                final_display_content = "The AI provided a structured JSON response. See above for details."
            else:
                # If parsed, but not dict (e.g., a simple string JSON), display it directly
                st.subheader("💬 AI Response (Parsed):")
                st.write(parsed_data)
                final_display_content = str(parsed_data)


        except json.JSONDecodeError:
            # If not JSON, assume it's a natural language response
            parsing_successful = False
            # Check if it was a "No relevant docs" message, if so, that's already the final content
            if "No relevant documentation found" in full_response_content:
                st.subheader("⚠️ Information:")
                st.write(full_response_content) # Display the existing message
            else:
                st.subheader("💬 AI Response:")
                st.write(full_response_content)
            final_display_content = full_response_content # Store the raw content

        except Exception as e:
            parsing_successful = False
            parsing_error_message = f"An unexpected error occurred during parsing or display: {e}. Raw response:\n{full_response_content}"
            st.error(parsing_error_message) # Show error to user
            st.subheader("💬 AI Response (Fallback due to error):")
            st.write(full_response_content) # Still show raw response
            final_display_content = full_response_content

        # Update session state messages for history using the `final_display_content`
        # This appends the assistant's final rendered output to the chat history
        st.session_state.messages.append({"role": "assistant", "content": final_display_content})

        # Update search history
        st.session_state.search_history.insert(0, {
            'query': query_text,
            'device_type': final_device_type,
            'results_count': len(relevant_docs), # Will be 0 for direct answers/follow-ups
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_used': model_name_setting,
            'parsing_successful': parsing_successful
        })
        st.session_state.search_history = st.session_state.search_history[:10]

        if show_debug_setting:
            debug_info_content = {
                'query': query_text,
                'device_type': final_device_type,
                'num_docs_retrieved': len(relevant_docs),
                'model_used': model_name_setting,
                'prompt_type': "Conversational" if is_follow_up_question else "RAG-enhanced", # More descriptive
                'prompt_content': generated_rag_prompt if generated_rag_prompt else "N/A (conversational)", # Store actual prompt
                'min_relevance_threshold': min_relevance_setting,
                'parsing_successful': parsing_successful
            }
            if parsing_error_message:
                debug_info_content['parsing_error'] = parsing_error_message
            
            # Display debug info
            with st.expander("View Debug Info"):
                st.json(debug_info_content)
            
            # Append debug info to session messages as well, for history display if needed
            st.session_state.messages.append({"role": "assistant", "content": f"```json\n{json.dumps(debug_info_content, indent=2)}\n```"})


def rag_pipeline(
    query: str,
    device_type: str,
    model,
    mac_collection,
    iphone_collection,
    groq_client,
    chat_history: List[Dict],
    n_results: int = 5,
    min_relevance: float = 0.3,
    model_name: str = "llama-3.1-8b-instant",
    show_debug: bool = False
) -> Dict:
    """Complete RAG Pipeline that returns the streamed LLM response and relevant docs."""

    # Step 1: Retrieve relevant documents
    relevant_docs = retrieve_relevant_docs(query, device_type, model, mac_collection, iphone_collection, n_results, min_relevance, show_debug)

    if not relevant_docs:
        # Return a special indicator for no relevant documents
        return {
            'query': query,
            'device_type': device_type,
            'retrieved_docs': [],
            'prompt': '',
            'llm_response_generator': None, # Indicate no LLM call made
            'num_docs_used': 0,
            'error': 'No relevant documents found'
        }

    # Step 2: Create RAG-enhanced prompt with context from retrieved documents for the *current* turn
    rag_prompt_content = create_repair_prompt(query, relevant_docs, device_type)

    temp_chat_history_for_llm = list(chat_history[:-1]) + [{"role": "user", "content": rag_prompt_content}]

    # Step 3: Get LLM response (this returns the generator directly)
    llm_response_generator = get_llm_response(
        chat_history=temp_chat_history_for_llm,
        client=groq_client,
        device_type=device_type,
        model_name=model_name
    )

    # Return the generator and other relevant info. Parsing happens in process_user_query.
    return {
        'query': query,
        'device_type': device_type,
        'retrieved_docs': relevant_docs,
        'prompt': rag_prompt_content,
        'llm_response_generator': llm_response_generator, # Pass the generator
        'num_docs_used': len(relevant_docs),
        'error': None # No error at this stage, potential errors handled during streaming/parsing
    }

def is_inappropriate_query(query: str) -> bool:
    """
    Checks if a query contains inappropriate content based on a keyword list.
    A more advanced implementation would use a small classifier.
    """
    inappropriate_keywords = [
        "violence", "harm", "kill", "drug", "sex", "porn", "hate", "racist",
        "illegal", "weapon", "bomb", "exploit", "abuse", "cyberattack",
        "self-harm", "suicide", "threat", "harass", "discriminate", "malware"
    ]
    query_lower = query.lower()
    for keyword in inappropriate_keywords:
        if keyword in query_lower:
            return True
    return False

def main():
    # Initialize session state variables if they don't exist
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    if 'groq_api_key' not in st.session_state:
        st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")
    if 'device_type' not in st.session_state:
        st.session_state.device_type = 'Both' # Changed default to 'Both'
    if 'auto_detect_device' not in st.session_state:
        st.session_state.auto_detect_device = False
    if 'n_results' not in st.session_state:
        st.session_state.n_results = 5
    if 'min_relevance' not in st.session_state:
        st.session_state.min_relevance = 0.3
    if 'model_name' not in st.session_state:
        st.session_state.model_name = 'llama-3.1-8b-instant'
    if 'temperature' not in st.session_state: # Make sure this is initialized too
        st.session_state.temperature = 0.3
    if 'max_tokens' not in st.session_state: # Make sure this is initialized too
        st.session_state.max_tokens = 1500
    if 'top_p' not in st.session_state: # Make sure this is initialized too
        st.session_state.top_p = 1.0
    if 'show_debug_info' not in st.session_state:
        st.session_state.show_debug_info = False
    if 'llm_timeout' not in st.session_state:
        st.session_state.llm_timeout = 60.0 
    # Add this line to initialize chat_input_key:
    if 'chat_input_key' not in st.session_state:
        st.session_state.chat_input_key = 0
    # Initialize session state
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    # Initialize query in session state for persistence
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""

    # Initialize chat messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Apple Repair Assistant</h1>
        <p>Get expert repair guidance powered by AI and comprehensive repair documentation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load ChromaDB
    chroma_client, model, mac_collection, iphone_collection, error = load_chromadb()
    
    # Get statistics
    stats = get_collection_stats()
    
    # --- Ensure Groq client is initialized HERE, ONCE ---
    groq_client = setup_groq_client(st.session_state.groq_api_key)

    # Check if client initialized successfully
    if groq_client is None:
        st.warning("Groq client could not be initialized. Please ensure your Groq API key is provided and valid in the 'Settings' tab.")
        # Do NOT st.stop() here, allow the app to display settings for API key input.
    # --- End of Groq client initialization ---

    # Display stats or error (existing code)
    if stats.get("error") or error:
        st.error(f"Database connection error: {stats.get('error', error)}")
        st.info("Please ensure your ChromaDB collections are properly set up.")
        
        with st.expander("🔧 Debug Information", expanded=True):
            debug_info = debug_collections()
            st.json(debug_info)
        # Do not return here to allow the rest of the UI to load
        
        return
    
    # Stats display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📱 iPhone Guides", f"{stats['iphone_count']:,}")
    with col2:
        st.metric("💻 Mac Guides", f"{stats['mac_count']:,}")
    with col3:
        st.metric("📚 Total Chunks", f"{stats['total_count']:,}")
    
    st.divider()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔑 API Configuration")
        # This text_input updates st.session_state.groq_api_key
        st.session_state.groq_api_key = st.text_input(
            "Groq API Key",
            value=st.session_state.groq_api_key,
            type="password",
            help="Get your API key from console.groq.com. It's stored securely in your browser session."
        )

        # Your existing feedback for API key input
        if st.session_state.groq_api_key:
            st.success("✅ Groq API Key provided!") # This green message will appear
        else:
            st.warning("⚠️ Please enter your Groq API key to use the AI assistant.")
        groq_client = setup_groq_client(st.session_state.groq_api_key)
        
        # Model selection
        st.subheader("🤖 Model Settings")
        model_name = st.selectbox(
            "Groq Model:",
            [
                "llama-3.1-8b-instant",    # Fast + high quality
                "llama3-70b-8192",         # Very strong general model
            ],
            index=0,
            help="Different models offer varying speed/quality tradeoffs"
        )

        # Advanced search options
        st.subheader("🎛️ Search Options")
        
        # Modified device_type selectbox
        device_type = st.selectbox(
            "Device Type:",
            ["Both", "Mac", "iPhone"], # Added "Both" option
            index=0, # Set default to "Both"
            help="Choose the device you need help with"
        )
        
        n_results = st.slider(
            "Documents to retrieve:",
            min_value=1,
            max_value=10,
            value=5,
            help="More documents provide more context but may exceed token limits"
        )
        
        min_relevance = st.slider(
            "Minimum Relevance:", 
            0.1, 0.8, 0.3, 0.1,
            help="Higher values return more relevant but fewer results"
        )
        
        auto_detect = st.checkbox("Auto-detect device type", value=False)
        show_debug = st.checkbox("Show debug info", value=False)
        
        st.sidebar.subheader("Conversation Options")
        # Clear History Option
        if st.sidebar.button("Clear Chat History", key="clear_chat_button"):
            st.session_state.messages = []  # Clear the history
            st.info("Chat history cleared.")
            st.rerun()  # Rerun the app to update the UI and clear displayed messages

        # Search history in sidebar
        if st.session_state.search_history:
            st.subheader("📋 Recent Searches")
            for i, search in enumerate(st.session_state.search_history[:5]):
                with st.expander(f"{search['query'][:30]}..."):
                    st.write(f"**Device:** {search['device_type']}")
                    st.write(f"**Results:** {search['results_count']}")
                    st.write(f"**Time:** {search['timestamp']}")
    
    # Main content in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 AI Assistant", "📊 Analytics", "⚙️ Settings", "🔧 Debug"])
    
    with tab1:
        st.subheader("💬 Ask Your Repair Question")
      
        # Display chat history from session state
        for message in st.session_state.messages:
          with st.chat_message(message["role"]):
              st.markdown(message["content"])

        # Main chat input
        query = st.chat_input(
            "Describe your repair issue:",
            key=f"chat_input_main_{st.session_state.chat_input_key}", # Unique key for each turn
        )
        if query:
          # --- CALL process_user_query HERE ---
          process_user_query(
              query_text=query,
              device_type_setting=device_type, # From sidebar
              auto_detect_setting=auto_detect, # From sidebar
              model_embedding=model,
              mac_collection_db=mac_collection,
              iphone_collection_db=iphone_collection,
              groq_client_obj=groq_client, # *** IMPORTANT: Pass the initialized groq_client here ***
              n_results_setting=n_results, # From sidebar
              min_relevance_setting=min_relevance, # From sidebar
              model_name_setting=model_name, # From sidebar
              show_debug_setting=show_debug # From sidebar
          )
          st.session_state.chat_input_key += 1 # Increment for next input
          st.rerun()

        # Quick examples (Conditional rendering based on device_type)
        col1, col2 = st.columns(2)

        # Show Mac examples if device_type is 'Mac' or 'Both'
        if device_type == "Mac" or device_type == "Both":
            with col1:
                st.markdown("**Mac Examples:**")
                if st.button("🔋 Battery replacement steps", key="mac_battery_btn"):
                    process_user_query(
                        "MacBook Pro battery replacement", device_type, auto_detect, model,
                        mac_collection, iphone_collection, groq_client, n_results,
                        min_relevance, model_name, show_debug
                    )
                    st.session_state.chat_input_key += 1 # Increment for next input
                    st.rerun()

                if st.button("🖥️ Screen flickering problems", key="mac_screen_btn"):
                    process_user_query(
                        "Mac screen replacement", device_type, auto_detect, model,
                        mac_collection, iphone_collection, groq_client, n_results,
                        min_relevance, model_name, show_debug
                    )
                    st.session_state.chat_input_key += 1 # Increment for next input
                    st.rerun()
                    
                if st.button("⌨️ Keyboard not responding", key="mac_keyboard_btn"):
                    process_user_query(
                        "MacBook keyboard replacement", device_type, auto_detect, model,
                        mac_collection, iphone_collection, groq_client, n_results,
                        min_relevance, model_name, show_debug
                    )
                    st.session_state.chat_input_key += 1 # Increment for next input
                    st.rerun()
        
        # Show iPhone examples if device_type is 'iPhone' or 'Both'
        if device_type == "iPhone" or device_type == "Both":
            with col2:
                st.markdown("**iPhone Examples:**")
                if st.button("📱 Cracked screen repair", key="iphone_screen_btn"):
                    process_user_query(
                        "iPhone screen replacement", device_type, auto_detect, model,
                        mac_collection, iphone_collection, groq_client, n_results,
                        min_relevance, model_name, show_debug
                    )
                    st.session_state.chat_input_key += 1 # Increment for next input
                    st.rerun()
                    
                if st.button("🔋 Battery draining fast", key="iphone_battery_btn"):
                    process_user_query(
                        "iPhone battery replacement", device_type, auto_detect, model,
                        mac_collection, iphone_collection, groq_client, n_results,
                        min_relevance, model_name, show_debug
                    )
                    st.session_state.chat_input_key += 1 # Increment for next input
                    st.rerun()
                    
                if st.button("📷 Camera not working", key="iphone_camera_btn"):
                    process_user_query(
                        "iPhone camera problems", device_type, auto_detect, model,
                        mac_collection, iphone_collection, groq_client, n_results,
                        min_relevance, model_name, show_debug
                    )
                    st.session_state.chat_input_key += 1 # Increment for next input
                    st.rerun()
                

    with tab2:
        st.subheader("📊 Database Analytics")
        
        if not stats.get("error"):
            # Database Overview Section
            st.markdown("### 📈 Database Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Documents", 
                    f"{stats['total_count']:,}",
                    help="Total number of repair guide chunks in the database"
                )
            
            with col2:
                if stats['total_count'] > 0:
                    mac_percentage = (stats['mac_count'] / stats['total_count']) * 100
                    st.metric(
                        "Mac Coverage",
                        f"{mac_percentage:.1f}%",
                        help="Percentage of Mac repair guides"
                    )
                else:
                    st.metric("Mac Coverage", "0%")
            
            with col3:
                if stats['total_count'] > 0:
                    iphone_percentage = (stats['iphone_count'] / stats['total_count']) * 100
                    st.metric(
                        "iPhone Coverage",
                        f"{iphone_percentage:.1f}%",
                        help="Percentage of iPhone repair guides"
                    )
                else:
                    st.metric("iPhone Coverage", "0%")
            
            with col4:
                # Calculate search success rate if we have search history
                if st.session_state.search_history:
                    successful_searches = sum(1 for search in st.session_state.search_history if search['results_count'] > 0)
                    success_rate = (successful_searches / len(st.session_state.search_history)) * 100
                    st.metric(
                        "Search Success Rate",
                        f"{success_rate:.1f}%",
                        help="Percentage of searches that returned results"
                    )
                else:
                    st.metric("Search Success Rate", "N/A")
            
            st.divider()
            
            # Visual Analytics Section
            if stats['total_count'] > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart of device distribution
                    fig_pie = px.pie(
                        values=[stats['mac_count'], stats['iphone_count']],
                        names=['Mac Repairs', 'iPhone Repairs'],
                        title="📱 Repair Guide Distribution by Device Type",
                        color_discrete_map={
                            'Mac Repairs': '#1976d2', 
                            'iPhone Repairs': '#7b1fa2'
                        },
                        hole=0.4  # Donut chart
                    )
                    fig_pie.update_traces(
                        textposition='inside', 
                        textinfo='percent+label',
                        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                    )
                    fig_pie.update_layout(
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Bar chart comparison
                    fig_bar = px.bar(
                        x=['Mac Repairs', 'iPhone Repairs'],
                        y=[stats['mac_count'], stats['iphone_count']],
                        title="📊 Document Count Comparison",
                        color=['Mac Repairs', 'iPhone Repairs'],
                        color_discrete_map={
                            'Mac Repairs': '#1976d2', 
                            'iPhone Repairs': '#7b1fa2'
                        }
                    )
                    fig_bar.update_layout(
                        xaxis_title="Device Type",
                        yaxis_title="Number of Documents",
                        showlegend=False
                    )
                    fig_bar.update_traces(
                        hovertemplate='<b>%{x}</b><br>Documents: %{y:,}<extra></extra>'
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            st.divider()
            
            # Search Analytics Section
            if st.session_state.search_history:
                st.markdown("### 🔍 Search Analytics")
                
                # Search statistics
                total_searches = len(st.session_state.search_history)
                successful_searches = sum(1 for search in st.session_state.search_history if search['results_count'] > 0)
                avg_results = sum(search['results_count'] for search in st.session_state.search_history) / total_searches if total_searches > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Searches", total_searches)
                
                with col2:
                    st.metric("Successful Searches", successful_searches)
                
                with col3:
                    success_rate = (successful_searches / total_searches) * 100 if total_searches > 0 else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                with col4:
                    st.metric("Avg Results/Search", f"{avg_results:.1f}")
                
                # Device type and model usage analytics
                col1, col2 = st.columns(2)
                
                with col1:
                    # Device type distribution in searches
                    device_counts = {}
                    for search in st.session_state.search_history:
                        device = search['device_type']
                        device_counts[device] = device_counts.get(device, 0) + 1
                    
                    if device_counts:
                        fig_device = px.bar(
                            x=list(device_counts.keys()),
                            y=list(device_counts.values()),
                            title="🔍 Search Queries by Device Type",
                            color=list(device_counts.keys()),
                            color_discrete_map={'Mac': '#1976d2', 'iPhone': '#7b1fa2'}
                        )
                        fig_device.update_layout(
                            xaxis_title="Device Type",
                            yaxis_title="Number of Searches",
                            showlegend=False
                        )
                        fig_device.update_traces(
                            hovertemplate='<b>%{x}</b><br>Searches: %{y}<extra></extra>'
                        )
                        st.plotly_chart(fig_device, use_container_width=True)
                
                with col2:
                    # Model usage distribution
                    model_counts = {}
                    for search in st.session_state.search_history:
                        model = search.get('model_used', 'Unknown')
                        model_counts[model] = model_counts.get(model, 0) + 1
                    
                    if model_counts:
                        # Truncate long model names for better display
                        display_names = {}
                        for model in model_counts.keys():
                            if len(model) > 20:
                                display_names[model] = model[:17] + "..."
                            else:
                                display_names[model] = model
                        
                        fig_models = px.bar(
                            x=[display_names[k] for k in model_counts.keys()],
                            y=list(model_counts.values()),
                            title="🤖 AI Models Usage",
                            color=list(model_counts.values()),
                            color_continuous_scale='viridis'
                        )
                        fig_models.update_layout(
                            xaxis_title="Model",
                            yaxis_title="Usage Count",
                            showlegend=False,
                            xaxis_tickangle=-45
                        )
                        fig_models.update_traces(
                            hovertemplate='<b>%{x}</b><br>Used: %{y} times<extra></extra>'
                        )
                        st.plotly_chart(fig_models, use_container_width=True)
                
                # Search timeline
                if len(st.session_state.search_history) > 1:
                    st.markdown("### 📅 Search Activity Timeline")
                    
                    # Convert search history to DataFrame for timeline analysis
                    search_df = pd.DataFrame(st.session_state.search_history)
                    search_df['timestamp'] = pd.to_datetime(search_df['timestamp'])
                    search_df['date'] = search_df['timestamp'].dt.date
                    
                    # Group by date for timeline
                    daily_searches = search_df.groupby('date').size().reset_index(name='searches')
                    daily_searches['date'] = pd.to_datetime(daily_searches['date'])
                    
                    fig_timeline = px.line(
                        daily_searches,
                        x='date',
                        y='searches',
                        title="📈 Daily Search Activity",
                        markers=True
                    )
                    fig_timeline.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Number of Searches"
                    )
                    fig_timeline.update_traces(
                        hovertemplate='<b>%{x}</b><br>Searches: %{y}<extra></extra>'
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Results distribution
                st.markdown("### 📊 Search Results Distribution")
                
                result_counts = {}
                for search in st.session_state.search_history:
                    count = search['results_count']
                    if count == 0:
                        key = "No Results"
                    elif count <= 2:
                        key = "1-2 Results"
                    elif count <= 5:
                        key = "3-5 Results"
                    else:
                        key = "5+ Results"
                    result_counts[key] = result_counts.get(key, 0) + 1
                
                if result_counts:
                    fig_results = px.pie(
                        values=list(result_counts.values()),
                        names=list(result_counts.keys()),
                        title="🎯 Search Results Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_results.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                    )
                    st.plotly_chart(fig_results, use_container_width=True)
                
                st.divider()
                
                # Detailed search history table
                st.markdown("### 📋 Detailed Search History")
                
                # Create a more detailed DataFrame
                detailed_history = []
                for i, search in enumerate(st.session_state.search_history):
                    detailed_history.append({
                        'Search #': len(st.session_state.search_history) - i,
                        'Query': search['query'][:50] + ('...' if len(search['query']) > 50 else ''),
                        'Device': search['device_type'],
                        'Results Found': search['results_count'],
                        'Model Used': search.get('model_used', 'Unknown')[:20] + ('...' if len(search.get('model_used', '')) > 20 else ''),
                        'Timestamp': search['timestamp'],
                        'Status': '✅ Success' if search['results_count'] > 0 else '❌ No Results'
                    })
                
                history_df = pd.DataFrame(detailed_history)
                
                # Add filters
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    device_filter = st.selectbox(
                        "Filter by Device:",
                        ['All'] + list(history_df['Device'].unique()),
                        key="device_filter"
                    )
                
                with col2:
                    status_filter = st.selectbox(
                        "Filter by Status:",
                        ['All', '✅ Success', '❌ No Results'],
                        key="status_filter"
                    )
                
                with col3:
                    show_count = st.selectbox(
                        "Show:",
                        ['All', 'Last 5', 'Last 10'],
                        key="show_count"
                    )
                
                # Apply filters
                filtered_df = history_df.copy()
                
                if device_filter != 'All':
                    filtered_df = filtered_df[filtered_df['Device'] == device_filter]
                
                if status_filter != 'All':
                    filtered_df = filtered_df[filtered_df['Status'] == status_filter]
                
                if show_count == 'Last 5':
                    filtered_df = filtered_df.head(5)
                elif show_count == 'Last 10':
                    filtered_df = filtered_df.head(10)
                
                # Display the filtered table
                if not filtered_df.empty:
                    st.dataframe(
                        filtered_df,
                        use_container_width=True,
                        hide_index=True,
                        height=400
                    )
                    
                    # Export option
                    if st.button("📥 Export Search History as CSV"):
                        csv = filtered_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"search_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("No search history matches the selected filters.")
                
                # Clear history option
                st.divider()
                if st.button("🗑️ Clear Search History", type="secondary"):
                    if st.button("⚠️ Confirm Clear History", type="primary"):
                        st.session_state.search_history = []
                        st.success("Search history cleared!")
                        st.rerun()
            
            else:
                st.info("📭 No search history available yet. Start by asking a repair question in the AI Assistant tab!")
                
                # Show some sample queries to get started
                st.markdown("### 💡 Try These Sample Queries")
                sample_queries = [
                    "MacBook Pro battery replacement steps",
                    "iPhone screen not responding to touch",
                    "Mac won't boot up troubleshooting",
                    "iPhone camera app crashing",
                    "MacBook Air keyboard keys sticking",
                    "iPhone charging port not working"
                ]
                
                cols = st.columns(2)
                for i, query_example in enumerate(sample_queries): # Renamed 'query' to 'query_example' to avoid conflict
                    with cols[i % 2]:
                        st.code(query_example, language=None)
        
        else:
            st.error(f"❌ Cannot display analytics due to database error: {stats.get('error')}")
            st.info("Please check your ChromaDB setup and ensure the collections are properly initialized.")
      
    with tab3: 
        st.subheader("⚙️ System Settings & Configuration")
        
        # API Configuration Section
        st.markdown("### 🔑 API Configuration")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Groq API Settings
            st.markdown("#### Groq API Settings")
            
            current_api_key = st.session_state.get('groq_api_key', '')
            api_key_display = current_api_key[:8] + "..." + current_api_key[-4:] if len(current_api_key) > 12 else current_api_key
            
            if current_api_key:
                st.success(f"✅ API Key configured: {api_key_display}")
                
                # Test API connection
                if st.button("🧪 Test API Connection"):
                    groq_client = setup_groq_client(st.session_state.groq_api_key) # Pass API key
                    if groq_client:
                        try:
                            with st.spinner("Testing connection..."):
                                test_response = groq_client.chat.completions.create(
                                    model="llama-3.1-8b-instant",
                                    messages=[{"role": "user", "content": "Hello, respond with 'API connection successful'"}],
                                    max_tokens=50
                                )
                                if test_response.choices[0].message.content:
                                    st.success("🎉 API connection successful!")
                                else:
                                    st.error("❌ API responded but with empty content")
                        except Exception as e:
                            st.error(f"❌ API connection failed: {str(e)}")
                    else:
                        st.error("❌ Failed to initialize Groq client")
                
                # Option to clear API key
                if st.button("🗑️ Clear API Key", type="secondary"):
                    st.session_state['groq_api_key'] = ''
                    st.success("API key cleared")
                    st.rerun()
            else:
                st.warning("⚠️ No API key configured")
                st.info("Get your free API key from: https://console.groq.com/keys")
            
            # New API key input
            new_api_key = st.text_input(
                "Enter/Update Groq API Key:",
                type="password",
                placeholder="gsk_...",
                help="Your Groq API key for accessing AI models"
            )
            
            if new_api_key and new_api_key != current_api_key:
                if st.button("💾 Save API Key"):
                    st.session_state['groq_api_key'] = new_api_key
                    st.success("✅ API key saved!")
                    st.rerun()
        
        with col2:
            # API Usage Info Box
            st.markdown("#### 📊 API Info")
            st.info("""
            **Groq Models Available:**
            - llama-3.1-8b-instant (Recommended)
            - llama3-70b-8192
            
            **Rate Limits:**
            - Free tier: 30 requests/min
            - Check console.groq.com for details
            """)
        
        st.divider()
        
        # Model Configuration Section
        st.markdown("### 🤖 AI Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Default model selection
            current_model = st.session_state.get('default_model', 'llama-3.1-8b-instant')
            
            default_model = st.selectbox(
                "Default AI Model:",
                [
                    "llama-3.1-8b-instant",
                    "llama3-70b-8192"
                ],
                index=0 if current_model == 'llama-3.1-8b-instant' else 
                      1 if current_model == 'llama3-70b-8192' else 2,
                help="Choose the default model for AI responses"
            )
            
            if default_model != current_model:
                st.session_state['default_model'] = default_model
                st.success(f"Default model updated to: {default_model}")
            
            # Temperature setting
            temperature = st.slider(
                "Response Temperature:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('temperature', 0.3),
                step=0.1,
                help="Lower = more focused, Higher = more creative"
            )
            st.session_state['temperature'] = temperature
            
            # Max tokens setting
            max_tokens = st.slider(
                "Max Response Tokens:",
                min_value=500,
                max_value=2000,
                value=st.session_state.get('max_tokens', 1500),
                step=100,
                help="Maximum length of AI responses"
            )
            st.session_state['max_tokens'] = max_tokens
        
        with col2:
            # Model comparison info
            st.markdown("#### 🏆 Model Comparison")
            
            model_info = {
                "llama-3.1-8b-instant": {"speed": "⚡⚡⚡", "quality": "⭐⭐⭐⭐", "use_case": "General purpose, fast"},
                "llama3-70b-8192": {"speed": "⚡⚡", "quality": "⭐⭐⭐⭐⭐", "use_case": "Best quality, slower"}
            }
            
            for model, info in model_info.items():
                if model == default_model:
                    st.markdown(f"**🎯 {model}** (Selected)")
                else:
                    st.markdown(f"**{model}**")
                st.write(f"Speed: {info['speed']} | Quality: {info['quality']}")
                st.write(f"*{info['use_case']}*")
                st.write("")
        
        st.divider()
        
        # Search Configuration Section
        st.markdown("### 🔍 Search & Retrieval Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Default search parameters
            st.markdown("#### Search Parameters")
            
            default_n_results = st.slider(
                "Default Documents to Retrieve:",
                min_value=1,
                max_value=15,
                value=st.session_state.get('default_n_results', 5),
                help="Number of documents to retrieve by default"
            )
            st.session_state['default_n_results'] = default_n_results
            
            default_min_relevance = st.slider(
                "Default Minimum Relevance:",
                min_value=0.1,
                max_value=0.9,
                value=st.session_state.get('default_min_relevance', 0.3),
                step=0.05,
                help="Minimum similarity score for documents"
            )
            st.session_state['default_min_relevance'] = default_min_relevance
            
            # Auto-detection settings
            auto_detect_default = st.checkbox(
                "Enable Auto-Detection by Default",
                value=st.session_state.get('auto_detect_default', False),
                help="Automatically detect device type from queries"
            )
            st.session_state['auto_detect_default'] = auto_detect_default
            
            # Debug mode default
            debug_mode_default = st.checkbox(
                "Enable Debug Mode by Default",
                value=st.session_state.get('debug_mode_default', False),
                help="Show debug information by default"
            )
            st.session_state['debug_mode_default'] = debug_mode_default
        
        with col2:
            # Search optimization settings
            st.markdown("#### Optimization Settings")
            
            # Embedding model info (read-only for now)
            st.info("""
            **Current Embedding Model:**
            `all-mpnet-base-v2`
            
            **Model Details:**
            - Dimension: 768
            - Max Sequence: 514 tokens
            - Performance: High quality embeddings
            """)
            
            # Cache settings
            enable_caching = st.checkbox(
                "Enable Response Caching",
                value=st.session_state.get('enable_caching', True),
                help="Cache similar queries to improve response time"
            )
            st.session_state['enable_caching'] = enable_caching
            
            # Search timeout
            search_timeout = st.slider(
                "Search Timeout (seconds):",
                min_value=5,
                max_value=60,
                value=st.session_state.get('search_timeout', 30),
                step=5,
                help="Maximum time to wait for search results"
            )
            st.session_state['search_timeout'] = search_timeout
        
        st.divider()
        
        # Database Configuration Section
        st.markdown("### 🗄️ Database Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ChromaDB Settings")
            
            # Database path info
            db_path = "./chroma_db"
            st.code(f"Database Path: {db_path}")
            
            # Database status
            if os.path.exists(db_path):
                st.success("✅ Database directory exists")
                
                # Get database size
                try:
                    total_size = 0
                    for dirpath, dirnames, filenames in os.walk(db_path):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            total_size += os.path.getsize(filepath)
                    
                    size_mb = total_size / (1024 * 1024)
                    st.info(f"📊 Database Size: {size_mb:.2f} MB")
                except Exception as e:
                    st.warning(f"Could not calculate database size: {e}")
            else:
                st.error("❌ Database directory not found")
            
            # Collection health check
            if st.button("🏥 Check Database Health"):
                with st.spinner("Checking database health..."):
                    debug_info = debug_collections()
                    
                    if debug_info.get('error'):
                        st.error(f"Database Error: {debug_info['error']}")
                    else:
                        st.success("✅ Database health check passed")
                        
                        # Show collection details
                        for collection in ['mac_repairs', 'iphone_repairs']:
                            if f"{collection}_count" in debug_info:
                                count = debug_info[f"{collection}_count"]
                                st.write(f" {collection}: {count:,} documents")
                            else:
                                st.warning(f"⚠️ {collection}: Not found")
        
        with col2:
            st.markdown("#### Maintenance Operations")
            
            # Backup/Export options
            st.warning("🚧 **Maintenance Features**")
            st.info("""
            The following features would require additional implementation:
            
            - Database backup/restore
            - Collection rebuild
            - Index optimization
            - Data export/import
            """)
            
            # Placeholder buttons (would need implementation)
            if st.button("💾 Backup Database", disabled=True):
                st.info("Feature not implemented yet")
            
            if st.button("🔄 Rebuild Collections", disabled=True):
                st.info("Feature not implemented yet")
            
            if st.button("⚡ Optimize Indexes", disabled=True):
                st.info("Feature not implemented yet")
        
        st.divider()
        
        # UI/UX Settings Section
        st.markdown("### 🎨 Interface Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Display Preferences")
            
            # Results display options
            show_relevance_scores = st.checkbox(
                "Show Relevance Scores",
                value=st.session_state.get('show_relevance_scores', True),
                help="Display similarity scores for search results"
            )
            st.session_state['show_relevance_scores'] = show_relevance_scores
            
            show_document_ids = st.checkbox(
                "Show Document IDs",
                value=st.session_state.get('show_document_ids', False),
                help="Display internal document identifiers"
            )
            st.session_state['show_document_ids'] = show_document_ids
            
            compact_results = st.checkbox(
                "Compact Results View",
                value=st.session_state.get('compact_results', False),
                help="Show more results in less space"
            )
            st.session_state['compact_results'] = compact_results
            
            # Animation settings
            enable_animations = st.checkbox(
                "Enable UI Animations",
                value=st.session_state.get('enable_animations', True),
                help="Enable loading spinners and transitions"
            )
            st.session_state['enable_animations'] = enable_animations
        
        with col2:
            st.markdown("#### Content Settings")
            
            # Text truncation
            max_preview_length = st.slider(
                "Result Preview Length:",
                min_value=100,
                max_value=1000,
                value=st.session_state.get('max_preview_length', 300),
                step=50,
                help="Maximum characters to show in result previews"
            )
            st.session_state['max_preview_length'] = max_preview_length
            
            # History settings
            max_history_items = st.slider(
                "Max History Items:",
                min_value=5,
                max_value=50,
                value=st.session_state.get('max_history_items', 10),
                step=5,
                help="Maximum number of searches to keep in history"
            )
            st.session_state['max_history_items'] = max_history_items
            
            # Auto-clear history
            auto_clear_history = st.selectbox(
                "Auto-clear History:",
                ["Never", "After 24 hours", "After 7 days", "After 30 days"],
                index=0,
                help="Automatically clear old search history"
            )
            st.session_state['auto_clear_history'] = auto_clear_history
        
        st.divider()
        
        # Export/Import Settings Section
        st.markdown("### 📤 Export & Import Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Export Current Settings")
            
            # Prepare settings for export
            settings_to_export = {
                'default_model': st.session_state.get('default_model', 'llama-3.1-8b-instant'),
                'temperature': st.session_state.get('temperature', 0.3),
                'max_tokens': st.session_state.get('max_tokens', 1500),
                'default_n_results': st.session_state.get('default_n_results', 5),
                'default_min_relevance': st.session_state.get('default_min_relevance', 0.3),
                'auto_detect_default': st.session_state.get('auto_detect_default', False),
                'debug_mode_default': st.session_state.get('debug_mode_default', False),
                'enable_caching': st.session_state.get('enable_caching', True),
                'search_timeout': st.session_state.get('search_timeout', 30),
                'show_relevance_scores': st.session_state.get('show_relevance_scores', True),
                'show_document_ids': st.session_state.get('show_document_ids', False),
                'compact_results': st.session_state.get('compact_results', False),
                'enable_animations': st.session_state.get('enable_animations', True),
                'max_preview_length': st.session_state.get('max_preview_length', 300),
                'max_history_items': st.session_state.get('max_history_items', 10),
                'auto_clear_history': st.session_state.get('auto_clear_history', 'Never'),
                'exported_at': datetime.now().isoformat()
            }
            
            settings_json = json.dumps(settings_to_export, indent=2)
            
            st.download_button(
                label="📥 Download Settings",
                data=settings_json,
                file_name=f"repair_assistant_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Download current settings as JSON file"
            )
        
        with col2:
            st.markdown("#### Import Settings")
            
            uploaded_settings = st.file_uploader(
                "Upload Settings File:",
                type=['json'],
                help="Upload a previously exported settings file"
            )
            
            if uploaded_settings is not None:
                try:
                    settings_content = json.loads(uploaded_settings.read())
                    
                    st.success("✅ Settings file loaded successfully")
                    
                    # Show preview of settings
                    with st.expander("📋 Preview Settings"):
                        st.json(settings_content)
                    
                    if st.button("💾 Apply Imported Settings"):
                        # Apply settings to session state
                        for key, value in settings_content.items():
                            if key != 'exported_at':  # Skip metadata
                                st.session_state[key] = value
                        
                        st.success("🎉 Settings imported and applied successfully!")
                        st.info("Some settings may require a page refresh to take full effect.")
                        
                        # Option to refresh
                        if st.button("🔄 Refresh Page"):
                            st.rerun()
                
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON file. Please check the file format.")
                except Exception as e:
                    st.error(f"❌ Error importing settings: {str(e)}")
        
        st.divider()
        
        # Reset Settings Section
        st.markdown("### 🔧 Reset Settings")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.warning("⚠️ **Reset Options**")
            st.write("Use these options to reset different parts of your configuration:")
            
            # Individual reset options
            if st.button("🔄 Reset Search Settings"):
                st.session_state['default_n_results'] = 5
                st.session_state['default_min_relevance'] = 0.3
                st.session_state['auto_detect_default'] = False
                st.session_state['debug_mode_default'] = False
                st.success("Search settings reset to defaults")
            
            if st.button("🤖 Reset AI Model Settings"):
                st.session_state['default_model'] = 'llama-3.1-8b-instant'
                st.session_state['temperature'] = 0.3
                st.session_state['max_tokens'] = 1500
                st.success("AI model settings reset to defaults")
            
            if st.button("🎨 Reset UI Settings"):
                st.session_state['show_relevance_scores'] = True
                st.session_state['show_document_ids'] = False
                st.session_state['compact_results'] = False
                st.session_state['enable_animations'] = True
                st.session_state['max_preview_length'] = 300
                st.session_state['max_history_items'] = 10
                st.session_state['auto_clear_history'] = 'Never'
                st.success("UI settings reset to defaults")
        
        with col2:
            st.markdown("#### ⚠️ Danger Zone")
            
            # Complete reset
            if st.button("🚨 Reset All Settings", type="secondary"):
                if st.button("⚠️ Confirm Complete Reset", type="primary"):
                    # Reset all settings except API key
                    api_key = st.session_state.get('groq_api_key', '')
                    search_history = st.session_state.get('search_history', [])
                    
                    # Clear session state
                    for key in list(st.session_state.keys()):
                        if key not in ['groq_api_key', 'search_history']:
                            del st.session_state[key]
                    
                    # Restore API key and search history
                    st.session_state['groq_api_key'] = api_key
                    st.session_state['search_history'] = search_history
                    
                    st.success("🎉 All settings reset to defaults!")
                    st.info("API key and search history preserved.")
                    
                    if st.button("🔄 Refresh Page"):
                        st.rerun()
        
        # Footer info
        st.divider()
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9em;">
            <p>💡 <strong>Tip:</strong> Settings are automatically saved in your browser session.<br>
            Export your settings to preserve them across sessions.</p>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.subheader("🔧 Debug Information and Health Checks")
        
        st.markdown("### 🗄️ ChromaDB Status")
        db_debug_info = debug_collections()
        st.json(db_debug_info)
        
        st.markdown("### 🔍 Model and Client Status")
        
        st.write(f"**Sentence Transformer Model Loaded:** {'Yes' if model else 'No'}")
        if not model:
            st.warning("Sentence Transformer model failed to load. Check dependencies.")
        
        st.write(f"**Groq Client Initialized:** {'Yes' if groq_client else 'No'}")
        if not groq_client: 
            st.warning("Groq client could not be initialized. Please ensure your Groq API key is provided and valid in the 'Settings' tab.")
        
        st.markdown("### 🚀 Session State (Raw)")
        with st.expander("View full Streamlit Session State"):
            st.json(st.session_state.to_dict())

if __name__ == "__main__":
    main()
