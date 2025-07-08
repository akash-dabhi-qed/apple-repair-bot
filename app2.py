from flask import Flask, request, render_template_string, jsonify, session
import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
from datetime import datetime
import uuid

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Initialize model and ChromaDB with PERSISTENT storage
model = SentenceTransformer('all-mpnet-base-v2')
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

# Initialize collections
try:
    mac_collection = chroma_client.get_collection("mac_repairs")
    iphone_collection = chroma_client.get_collection("iphone_repairs")
    print("✅ Collections loaded successfully!")
except Exception as e:
    print(f"❌ Error loading collections: {e}")
    mac_collection = None
    iphone_collection = None

def get_collection_stats():
    """Get statistics about the collections"""
    stats = {}
    try:
        if mac_collection:
            mac_count = mac_collection.count()
            stats['mac_count'] = mac_count
        else:
            stats['mac_count'] = 0
            
        if iphone_collection:
            iphone_count = iphone_collection.count()
            stats['iphone_count'] = iphone_count
        else:
            stats['iphone_count'] = 0
            
        stats['total_count'] = stats['mac_count'] + stats['iphone_count']
    except Exception as e:
        print(f"Error getting stats: {e}")
        stats = {'mac_count': 0, 'iphone_count': 0, 'total_count': 0}
    
    return stats

def search_repairs(query, device_type="both", top_k=5, min_relevance=0.3):
    """Enhanced search function with relevance filtering"""
    if not query.strip():
        return []
    
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
                if relevance_score >= min_relevance:  # Filter by minimum relevance
                    results.append({
                        'device': 'Mac',
                        'content': doc,
                        'relevance_score': relevance_score,
                        'rank': i + 1,
                        'distance': distance
                    })
        except Exception as e:
            print(f"Error searching Mac collection: {e}")
    
    # Search iPhone repairs
    if device_type in ["iphone", "both"] and iphone_collection:
        try:
            iphone_results = iphone_collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )
            
            for i, (doc, distance) in enumerate(zip(iphone_results['documents'][0], iphone_results['distances'][0])):
                relevance_score = 1 - distance
                if relevance_score >= min_relevance:  # Filter by minimum relevance
                    results.append({
                        'device': 'iPhone',
                        'content': doc,
                        'relevance_score': relevance_score,
                        'rank': i + 1,
                        'distance': distance
                    })
        except Exception as e:
            print(f"Error searching iPhone collection: {e}")
    
    # Sort by relevance and limit results
    if device_type == "both":
        results = sorted(results, key=lambda x: x['relevance_score'], reverse=True)[:top_k]
    
    return results

def detect_device_type(query):
    """Enhanced device type detection"""
    query_lower = query.lower()
    
    # Mac keywords
    mac_keywords = ['mac', 'imac', 'macbook', 'mac mini', 'mac pro', 'macbook air', 'macbook pro']
    # iPhone keywords
    iphone_keywords = ['iphone', 'ios', 'apple phone']
    
    mac_score = sum(1 for keyword in mac_keywords if keyword in query_lower)
    iphone_score = sum(1 for keyword in iphone_keywords if keyword in query_lower)
    
    if mac_score > iphone_score:
        return "mac"
    elif iphone_score > mac_score:
        return "iphone"
    else:
        return "both"

def save_search_history(query, device_type, results_count):
    """Save search history to session"""
    if 'search_history' not in session:
        session['search_history'] = []
    
    search_entry = {
        'id': str(uuid.uuid4())[:8],
        'query': query,
        'device_type': device_type,
        'results_count': results_count,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    session['search_history'].insert(0, search_entry)
    # Keep only last 10 searches
    session['search_history'] = session['search_history'][:10]

# Enhanced HTML template with modern styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔧 AI Repair Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 30px;
            text-align: center;
            position: relative;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 700;
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .stats {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
            font-size: 0.9rem;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-number {
            display: block;
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        .main-content {
            padding: 40px 30px;
        }
        
        .search-section {
            margin-bottom: 30px;
        }
        
        .search-form {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .search-input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e1e5e9;
            border-radius: 12px;
            font-size: 16px;
            transition: border-color 0.3s ease;
        }
        
        .search-input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .search-btn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        
        .search-btn:hover {
            transform: translateY(-2px);
        }
        
        .filters {
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .filter-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .filter-select {
            padding: 8px 12px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 14px;
        }
        
        .results-section {
            margin-top: 30px;
        }
        
        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .results-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #333;
        }
        
        .results-count {
            color: #666;
            font-size: 0.9rem;
        }
        
        .result-card {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .result-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .device-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .device-mac {
            background: #e3f2fd;
            color: #1976d2;
        }
        
        .device-iphone {
            background: #f3e5f5;
            color: #7b1fa2;
        }
        
        .relevance-score {
            color: #666;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .result-content {
            line-height: 1.6;
            color: #444;
        }
        
        .no-results {
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }
        
        .no-results-icon {
            font-size: 4rem;
            margin-bottom: 20px;
            opacity: 0.5;
        }
        
        .sidebar {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            margin-top: 30px;
        }
        
        .sidebar h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.1rem;
        }
        
        .history-item {
            padding: 8px 0;
            border-bottom: 1px solid #e9ecef;
            font-size: 0.9rem;
        }
        
        .history-query {
            color: #667eea;
            font-weight: 500;
        }
        
        .history-meta {
            color: #666;
            font-size: 0.8rem;
            margin-top: 2px;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
            .search-form {
                flex-direction: column;
            }
            
            .filters {
                justify-content: center;
            }
            
            .results-header {
                flex-direction: column;
                gap: 10px;
                text-align: center;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔧 AI Repair Assistant</h1>
            <p>Get expert repair guidance for Mac and iPhone devices</p>
            <div class="stats">
                <div class="stat-item">
                    <span class="stat-number">{{ stats.mac_count }}</span>
                    <span>Mac Guides</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{{ stats.iphone_count }}</span>
                    <span>iPhone Guides</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{{ stats.total_count }}</span>
                    <span>Total Chunks</span>
                </div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="search-section">
                <form method="POST" class="search-form" id="searchForm">
                    <input type="text" 
                           name="query" 
                           class="search-input"
                           placeholder="Describe your repair issue (e.g., 'iPhone 12 screen replacement')"
                           value="{{ query or '' }}" 
                           autofocus
                           required>
                    <button type="submit" class="search-btn">
                        🔍 Search
                    </button>
                </form>
                
                <div class="filters">
                    <div class="filter-group">
                        <label for="deviceType">Device:</label>
                        <select name="device_type" class="filter-select" form="searchForm">
                            <option value="both" {{ 'selected' if device_type == 'both' else '' }}>All Devices</option>
                            <option value="mac" {{ 'selected' if device_type == 'mac' else '' }}>Mac Only</option>
                            <option value="iphone" {{ 'selected' if device_type == 'iphone' else '' }}>iPhone Only</option>
                        </select>
                    </div>
                    
                    <div class="filter-group">
                        <label for="resultCount">Results:</label>
                        <select name="top_k" class="filter-select" form="searchForm">
                            <option value="5" {{ 'selected' if top_k == 5 else '' }}>5 Results</option>
                            <option value="8" {{ 'selected' if top_k == 8 else '' }}>8 Results</option>
                            <option value="10" {{ 'selected' if top_k == 10 else '' }}>10 Results</option>
                        </select>
                    </div>
                    
                    <div class="filter-group">
                        <label for="minRelevance">Min Relevance:</label>
                        <select name="min_relevance" class="filter-select" form="searchForm">
                            <option value="0.2" {{ 'selected' if min_relevance == 0.2 else '' }}>20%</option>
                            <option value="0.3" {{ 'selected' if min_relevance == 0.3 else '' }}>30%</option>
                            <option value="0.4" {{ 'selected' if min_relevance == 0.4 else '' }}>40%</option>
                            <option value="0.5" {{ 'selected' if min_relevance == 0.5 else '' }}>50%</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Searching repair database...</p>
            </div>
            
            {% if query %}
                <div class="results-section">
                    <div class="results-header">
                        <h2 class="results-title">Search Results</h2>
                        <span class="results-count">{{ results|length }} results for "{{ query }}"</span>
                    </div>
                    
                    {% if results %}
                        {% for result in results %}
                            <div class="result-card">
                                <div class="result-header">
                                    <span class="device-badge device-{{ result.device.lower() }}">
                                        {{ result.device }} Repair
                                    </span>
                                    <span class="relevance-score">
                                        {{ "%.1f"|format(result.relevance_score * 100) }}% match
                                    </span>
                                </div>
                                <div class="result-content">
                                    {{ result.content }}
                                </div>
                            </div>
                        {% endfor %}
                    {% else %}
                        <div class="no-results">
                            <div class="no-results-icon">🔍</div>
                            <h3>No Results Found</h3>
                            <p>Try adjusting your search terms or lowering the minimum relevance threshold.</p>
                            <p><strong>Tips:</strong></p>
                            <ul style="text-align: left; display: inline-block; margin-top: 10px;">
                                <li>Use specific device models (e.g., "iPhone 12", "MacBook Pro")</li>
                                <li>Include the repair type (e.g., "screen", "battery", "keyboard")</li>
                                <li>Try broader terms if too specific</li>
                            </ul>
                        </div>
                    {% endif %}
                </div>
            {% endif %}
            
            {% if session.search_history %}
                <div class="sidebar">
                    <h3>📋 Recent Searches</h3>
                    {% for search in session.search_history[:5] %}
                        <div class="history-item">
                            <div class="history-query">{{ search.query }}</div>
                            <div class="history-meta">
                                {{ search.device_type }} • {{ search.results_count }} results • {{ search.timestamp }}
                            </div>
                        </div>
                    {% endfor %}
                </div>
            {% endif %}
        </div>
    </div>
    
    <script>
        document.getElementById('searchForm').addEventListener('submit', function() {
            document.getElementById('loading').style.display = 'block';
        });
        
        // Auto-submit form when filters change
        document.querySelectorAll('.filter-select').forEach(select => {
            select.addEventListener('change', function() {
                if (document.querySelector('.search-input').value.trim()) {
                    document.getElementById('searchForm').submit();
                }
            });
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    query = None
    results = []
    device_type = "both"
    top_k = 5
    min_relevance = 0.3
    
    # Get collection statistics
    stats = get_collection_stats()
    
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        device_type = request.form.get('device_type', 'both')
        top_k = int(request.form.get('top_k', 5))
        min_relevance = float(request.form.get('min_relevance', 0.3))
        
        if query:
            # Auto-detect device type if set to "both"
            if device_type == "both":
                detected_type = detect_device_type(query)
                if detected_type != "both":
                    device_type = detected_type
            
            # Perform search
            results = search_repairs(query, device_type=device_type, top_k=top_k, min_relevance=min_relevance)
            
            # Save to search history
            save_search_history(query, device_type, len(results))
    
    return render_template_string(HTML_TEMPLATE, 
                                query=query, 
                                results=results, 
                                device_type=device_type, 
                                top_k=top_k, 
                                min_relevance=min_relevance,
                                stats=stats)

@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint for programmatic access"""
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    query = data['query']
    device_type = data.get('device_type', 'both')
    top_k = data.get('top_k', 5)
    min_relevance = data.get('min_relevance', 0.3)
    
    results = search_repairs(query, device_type=device_type, top_k=top_k, min_relevance=min_relevance)
    
    return jsonify({
        'query': query,
        'device_type': device_type,
        'results_count': len(results),
        'results': results
    })

@app.route('/api/stats')
def api_stats():
    """API endpoint to get collection statistics"""
    stats = get_collection_stats()
    return jsonify(stats)

if __name__ == '__main__':
    print("🚀 Starting Enhanced Repair Assistant Web Interface...")
    print("📊 Collection Statistics:")
    stats = get_collection_stats()
    print(f"   Mac repairs: {stats['mac_count']:,}")
    print(f"   iPhone repairs: {stats['iphone_count']:,}")
    print(f"   Total chunks: {stats['total_count']:,}")
    print("🌐 Visit: http://localhost:5000")
    print("📡 API available at: http://localhost:5000/api/search")
    
    app.run(debug=True, port=5000, host='0.0.0.0')
