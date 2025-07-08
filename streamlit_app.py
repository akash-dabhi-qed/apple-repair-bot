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

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="🔧 AI Repair Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling (similar to your Next.js approach)
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
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_db():
    """Load model and database (cached for performance)"""
    try:
        # Load embedding model
        model = SentenceTransformer('all-mpnet-base-v2')
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get collections
        mac_collection = chroma_client.get_collection("mac_repairs")
        iphone_collection = chroma_client.get_collection("iphone_repairs")
        
        return model, mac_collection, iphone_collection, None
    except Exception as e:
        return None, None, None, str(e)

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
    """Enhanced search function"""
    model, mac_collection, iphone_collection, error = load_model_and_db()
    
    if error or not model:
        st.error(f"Database error: {error}")
        return []
    
    if not query.strip():
        return []
    
    # Generate query embedding
    query_embedding = model.encode([query])
    results = []
    
    # Search Mac repairs
    if device_type in ["mac", "both"] and mac_collection:
        try:
            mac_results = mac_collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )
            
            for i, (doc, distance) in enumerate(zip(mac_results['documents'][0], mac_results['distances'][0])):
                relevance_score = 1 - distance
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
    
    # Search iPhone repairs
    if device_type in ["iphone", "both"] and iphone_collection:
        try:
            iphone_results = iphone_collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )
            
            for i, (doc, distance) in enumerate(zip(iphone_results['documents'][0], iphone_results['distances'][0])):
                relevance_score = 1 - distance
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
    
    # Sort by relevance
    if device_type == "both":
        results = sorted(results, key=lambda x: x['relevance_score'], reverse=True)[:top_k]
    
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
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📊 Analytics", "⚙️ Settings"])
    
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
                st.write("• Lowering the minimum relevance threshold")
                st.write("• Including specific device models (e.g., 'iPhone 12', 'MacBook Pro')")
    
    with tab2:
        st.subheader("📊 Database Analytics")
        
        if not stats.get("error"):
            # Pie chart of device distribution
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

if __name__ == "__main__":
    main()
