from flask import Flask, request, render_template_string
import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Initialize model and ChromaDB with PERSISTENT storage (same as embedding.py)
model = SentenceTransformer('all-mpnet-base-v2')
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",  # Same path as in embedding.py
    settings=Settings(anonymized_telemetry=False)
)
mac_collection = chroma_client.get_collection("mac_repairs")
iphone_collection = chroma_client.get_collection("iphone_repairs")

def search_repairs(query, device_type="both", top_k=5):
    """Search function (same as in search_repairs.py)"""
    query_embedding = model.encode([query])
    results = []
    
    if device_type in ["mac", "both"]:
        mac_results = mac_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        for i, (doc, distance) in enumerate(zip(mac_results['documents'][0], mac_results['distances'][0])):
            results.append({
                'device': 'Mac',
                'content': doc,
                'relevance_score': 1 - distance,
                'rank': i + 1
            })
    
    if device_type in ["iphone", "both"]:
        iphone_results = iphone_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        for i, (doc, distance) in enumerate(zip(iphone_results['documents'][0], iphone_results['distances'][0])):
            results.append({
                'device': 'iPhone',
                'content': doc,
                'relevance_score': 1 - distance,
                'rank': i + 1
            })
    
    if device_type == "both":
        results = sorted(results, key=lambda x: x['relevance_score'], reverse=True)[:top_k]
    
    return results

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🔧 Repair Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .search-box { margin: 20px 0; }
        input[type="text"] { width: 70%; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; background: #007cba; color: white; border: none; cursor: pointer; }
        button:hover { background: #005a87; }
        .result { margin: 20px 0; padding: 15px; border-left: 4px solid #007cba; background: #f5f5f5; }
        .device { font-weight: bold; color: #007cba; }
        .score { color: #666; font-size: 12px; }
        .no-results { text-align: center; color: #666; margin: 40px 0; }
        .header { text-align: center; margin-bottom: 30px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔧 Repair Assistant</h1>
        <p>Ask me about Mac or iPhone repair issues!</p>
    </div>
    
    <form method="POST" class="search-box">
        <input type="text" name="query" placeholder="Describe your repair issue..." value="{{ query or '' }}" autofocus>
        <button type="submit">Search</button>
    </form>
    
    {% if query %}
        <h3>Results for: "{{ query }}"</h3>
        {% if results %}
            {% for result in results %}
                <div class="result">
                    <div class="device">{{ result.device }} Repair</div>
                    <div class="score">Relevance: {{ "%.2f"|format(result.relevance_score) }}</div>
                    <p>{{ result.content }}</p>
                </div>
            {% endfor %}
        {% else %}
            <div class="no-results">
                <p>No relevant repair information found for your query.</p>
                <p>Try rephrasing your question or using different keywords.</p>
            </div>
        {% endif %}
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    query = None
    results = []
    
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        
        if query:
            # Simple device type detection
            device_type = "both"
            if "mac" in query.lower() or "imac" in query.lower() or "macbook" in query.lower():
                device_type = "mac"
            elif "iphone" in query.lower():
                device_type = "iphone"
            
            results = search_repairs(query, device_type=device_type, top_k=5)
    
    return render_template_string(HTML_TEMPLATE, query=query, results=results)

if __name__ == '__main__':
    print("Starting Repair Assistant Web Interface...")
    print("Visit: http://localhost:5000")
    app.run(debug=True, port=5000)
