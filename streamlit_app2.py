import streamlit as st
import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import time
import os

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
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
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
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_db():
    """Load model and database (cached for performance)"""
    try:
        # Load embedding model
        model = SentenceTransformer('all-mpnet-base-v2')
        
        # Check if ChromaDB path exists
        db_path = "./chroma_db"
        if not os.path.exists(db_path):
            return None, None, None, f"ChromaDB path not found: {db_path}"
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
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
        
        return model, mac_collection, iphone_collection, None
    except Exception as e:
        return None, None, None, str(e)

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

@st.cache_data
def get_collection_stats():
    """Get collection statistics (cached)"""
    try:
        _, mac_collection, iphone_collection, error = load_model_and_db()
        
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

def search_repairs(query: str, device_type: str = "both", top_k: int = 5, min_relevance: float = 0.3) -> List[Dict]:
    """Enhanced search function with better error handling"""
    model, mac_collection, iphone_collection, error = load_model_and_db()
    
    if error or not model:
        st.error(f"Database error: {error}")
        return []
    
    if not query.strip():
        return []
    
    # Debug: Show what collections are available
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
        st.write(f"- Query embedding generated successfully: shape {query_embedding.shape}")
    except Exception as e:
        st.error(f"Error generating query embedding: {e}")
        return []
    
    results = []
    
    # Search Mac repairs
    if device_type in ["mac", "both"] and mac_collection:
        try:
            st.write("🔍 Searching Mac collection...")
            mac_results = mac_collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )
            
            st.write(f"- Mac query returned {len(mac_results['documents'][0])} results")
            
            for i, (doc, distance) in enumerate(zip(mac_results['documents'][0], mac_results['distances'][0])):
                relevance_score = 1 - distance
                st.write(f"  - Result {i+1}: relevance={relevance_score:.3f}, distance={distance:.3f}")
                
                if relevance_score >= min_relevance:
                    results.append({
                        'device': 'Mac',
                        'content': doc,
                        'relevance_score': relevance_score,
                        'rank': i + 1,
                        'distance': distance
                    })
        except Exception as e:
            st.error(f"Error searching Mac collection: {e}")
            st.write(f"Error details: {type(e).__name__}: {str(e)}")
    
    # Search iPhone repairs
    if device_type in ["iphone", "both"] and iphone_collection:
        try:
            st.write("🔍 Searching iPhone collection...")
            iphone_results = iphone_collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )
            
            st.write(f"- iPhone query returned {len(iphone_results['documents'][0])} results")
            
            for i, (doc, distance) in enumerate(zip(iphone_results['documents'][0], iphone_results['distances'][0])):
                relevance_score = 1 - distance
                st.write(f"  - Result {i+1}: relevance={relevance_score:.3f}, distance={distance:.3f}")
                
                if relevance_score >= min_relevance:
                    results.append({
                        'device': 'iPhone',
                        'content': doc,
                        'relevance_score': relevance_score,
                        'rank': i + 1,
                        'distance': distance
                    })
        except Exception as e:
            st.error(f"Error searching iPhone collection: {e}")
            st.write(f"Error details: {type(e).__name__}: {str(e)}")
    
    # Sort by relevance
    if device_type == "both" and results:
        results = sorted(results, key=lambda x: x['relevance_score'], reverse=True)[:top_k]
    
    st.write(f"🎯 **Final results:** {len(results)} results above threshold {min_relevance}")
    
    return results

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
    device_class = "device-badge-mac" if result['device'] == 'Mac' else "device-badge-iphone"
    relevance_class = get_relevance_class(result['relevance_score'])
    
    st.markdown(f"""
    <div class="result-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span class="{device_class}">{result['device']} Repair</span>
            <span class="{relevance_class}">{result['relevance_score']:.1%} match</span>
        </div>
        <p style="line-height: 1.6; color: #444; margin: 0;">{result['content']}</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Initialize session state
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔧 AI Repair Assistant</h1>
        <p>Get expert repair guidance for Mac and iPhone devices</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get statistics
    stats = get_collection_stats()
    
    # Display stats or error
    if stats.get("error"):
        st.error(f"Database connection error: {stats['error']}")
        st.info("Please ensure your ChromaDB collections are properly set up.")
        
        # Show debug information
        with st.expander("🔧 Debug Information", expanded=True):
            debug_info = debug_collections()
            st.json(debug_info)
        
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
    
    # Main content in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📊 Analytics", "⚙️ Settings", "🔧 Debug"])
    
    with tab1:
        # Search interface
        st.subheader("Search Repair Database")
        
        # Search form
        with st.form("search_form"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                query = st.text_input(
                    "Describe your repair issue:",
                    placeholder="e.g., iPhone 12 screen replacement, MacBook Pro keyboard repair",
                    help="Be specific about device model and issue type for best results"
                )
            
            with col2:
                search_button = st.form_submit_button("🔍 Search", use_container_width=True)
        
        # Advanced options in sidebar
        with st.sidebar:
            st.subheader("🎛️ Search Options")
            
            device_type = st.selectbox(
                "Device Type:",
                ["both", "mac", "iphone"],
                format_func=lambda x: {"both": "All Devices", "mac": "Mac Only", "iphone": "iPhone Only"}[x]
            )
            
            top_k = st.slider("Number of Results:", 3, 15, 5)
            min_relevance = st.slider("Minimum Relevance:", 0.1, 0.8, 0.3, 0.1)
            
            auto_detect = st.checkbox("Auto-detect device type", value=True)
            show_debug = st.checkbox("Show debug info", value=True)
            
            # Search history in sidebar
            if st.session_state.search_history:
                st.subheader("📋 Recent Searches")
                for i, search in enumerate(st.session_state.search_history[:5]):
                    with st.expander(f"{search['query'][:30]}..."):
                        st.write(f"**Device:** {search['device_type']}")
                        st.write(f"**Results:** {search['results_count']}")
                        st.write(f"**Time:** {search['timestamp']}")
        
        # Process search
        if search_button and query:
            # Auto-detect device type if enabled
            if auto_detect and device_type == "both":
                detected_type = detect_device_type(query)
                if detected_type != "both":
                    device_type = detected_type
                    st.info(f"Auto-detected device type: {device_type.title()}")
            
            # Show loading spinner
            with st.spinner("Searching repair database..."):
                if show_debug:
                    st.markdown('<div class="debug-section">', unsafe_allow_html=True)
                    results = search_repairs(query, device_type, top_k, min_relevance)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    results = search_repairs(query, device_type, top_k, min_relevance)
            
            # Save to history
            st.session_state.search_history.insert(0, {
                'query': query,
                'device_type': device_type,
                'results_count': len(results),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            # Keep only last 10 searches
            st.session_state.search_history = st.session_state.search_history[:10]
            
            # Display results
            if results:
                st.success(f"Found {len(results)} relevant repair guides")
                
                # Results summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_relevance = sum(r['relevance_score'] for r in results) / len(results)
                    st.metric("Avg. Relevance", f"{avg_relevance:.1%}")
                with col2:
                    mac_count = sum(1 for r in results if r['device'] == 'Mac')
                    st.metric("Mac Results", mac_count)
                with col3:
                    iphone_count = sum(1 for r in results if r['device'] == 'iPhone')
                    st.metric("iPhone Results", iphone_count)
                
                st.divider()
                
                # Display results
                for i, result in enumerate(results):
                    render_result_card(result, i)
                
            else:
                st.warning("No results found. Try:")
                st.write("• Using different keywords")
                st.write("• Lowering the minimum relevance threshold (currently set to {:.1%})".format(min_relevance))
                st.write("• Including specific device models (e.g., 'iPhone 12', 'MacBook Pro')")
                
                # Suggest some test queries
                st.info("**Try these test queries:**")
                st.write("• `battery replacement`")
                st.write("• `screen repair`")
                st.write("• `water damage`")
                st.write("• `charging port`")
    
    with tab2:
        st.subheader("📊 Database Analytics")
        
        if not stats.get("error"):
            # Pie chart of device distribution
            if stats['total_count'] > 0:
                fig_pie = px.pie(
                    values=[stats['mac_count'], stats['iphone_count']],
                    names=['Mac', 'iPhone'],
                    title="Repair Guide Distribution",
                    color_discrete_map={'Mac': '#1976d2', 'iPhone': '#7b1fa2'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Search history analytics
            if st.session_state.search_history:
                st.subheader("Search History Analytics")
                
                # Device type distribution in searches
                device_counts = {}
                for search in st.session_state.search_history:
                    device = search['device_type']
                    device_counts[device] = device_counts.get(device, 0) + 1
                
                if device_counts:
                    fig_bar = px.bar(
                        x=list(device_counts.keys()),
                        y=list(device_counts.values()),
                        title="Search Queries by Device Type",
                        labels={'x': 'Device Type', 'y': 'Number of Searches'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Recent search results table
                search_df = pd.DataFrame(st.session_state.search_history)
                st.subheader("Recent Search Results")
                st.dataframe(search_df, use_container_width=True)
        else:
            st.error("Cannot display analytics due to database connection issues.")
    
    with tab3:
        st.subheader("⚙️ System Settings")
        
        # Database information
        st.write("**Database Configuration:**")
        st.code("""
        Database Type: ChromaDB (Persistent)
        Embedding Model: all-mpnet-base-v2
        Database Path: ./chroma_db
        Collections: mac_repairs, iphone_repairs
        """)
        
        # Model information
        st.write("**Model Information:**")
        st.code("""
        Model: sentence-transformers/all-mpnet-base-v2
        Max Sequence Length: 384 tokens
        Embedding Dimension: 768
        Optimal Chunk Size: 300-350 tokens
        """)
        
        # Export options
        st.subheader("📤 Export Options")
        
        if st.button("Export Search History"):
            if st.session_state.search_history:
                df = pd.DataFrame(st.session_state.search_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"search_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No search history to export.")
        
        # Clear history
        if st.button("Clear Search History", type="secondary"):
            st.session_state.search_history = []
            st.success("Search history cleared!")
            st.rerun()
    
    with tab4:
        st.subheader("🔧 Debug Information")
        
        # Run debug check
        if st.button("🔍 Run Debug Check"):
            debug_info = debug_collections()
            st.json(debug_info)
        
        # Manual collection test
        st.subheader("Manual Collection Test")
        
        test_query = st.text_input("Test Query:", value="battery replacement")
        
        if st.button("Test Collections"):
            if test_query:
                model, mac_collection, iphone_collection, error = load_model_and_db()
                
                if error:
                    st.error(f"Error: {error}")
                else:
                    # Test Mac collection
                    if mac_collection:
                        try:
                            mac_test = mac_collection.query(
                                query_texts=[test_query],
                                n_results=3
                            )
                            st.write("**Mac Collection Test Results:**")
                            for i, (doc, distance) in enumerate(zip(mac_test['documents'][0], mac_test['distances'][0])):
                                st.write(f"{i+1}. Distance: {distance:.3f}")
                                st.write(f"   Content: {doc[:200]}...")
                                st.write("---")
                        except Exception as e:
                            st.error(f"Mac collection error: {e}")
                    
                    # Test iPhone collection
                    if iphone_collection:
                        try:
                            iphone_test = iphone_collection.query(
                                query_texts=[test_query],
                                n_results=3
                            )
                            st.write("**iPhone Collection Test Results:**")
                            for i, (doc, distance) in enumerate(zip(iphone_test['documents'][0], iphone_test['distances'][0])):
                                st.write(f"{i+1}. Distance: {distance:.3f}")
                                st.write(f"   Content: {doc[:200]}...")
                                st.write("---")
                        except Exception as e:
                            st.error(f"iPhone collection error: {e}")

if __name__ == "__main__":
    main()
