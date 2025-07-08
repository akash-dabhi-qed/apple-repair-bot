import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Initialize the same model used for embeddings
model = SentenceTransformer('all-mpnet-base-v2')

# Initialize Chroma client with PERSISTENT storage (same as embeddings script)
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

# Get existing collections
mac_collection = chroma_client.get_collection("mac_repairs")
iphone_collection = chroma_client.get_collection("iphone_repairs")

def search_repairs(query, device_type="both", top_k=5):
    """
    Search for repair information based on user query
    
    Args:
        query (str): User's question or problem description
        device_type (str): "mac", "iphone", or "both"
        top_k (int): Number of results to return
    
    Returns:
        list: Relevant repair information
    """
    # Encode the user's query
    query_embedding = model.encode([query])
    
    results = []
    
    # Search Mac repairs
    if device_type in ["mac", "both"]:
        mac_results = mac_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        for i, (doc, distance) in enumerate(zip(mac_results['documents'][0], mac_results['distances'][0])):
            results.append({
                'device': 'Mac',
                'content': doc,
                'relevance_score': 1 - distance,  # Convert distance to similarity score
                'rank': i + 1
            })
    
    # Search iPhone repairs
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
    
    # Sort by relevance score if searching both devices
    if device_type == "both":
        results = sorted(results, key=lambda x: x['relevance_score'], reverse=True)[:top_k]
    
    return results

def format_results(results):
    """Format search results for display"""
    if not results:
        return "No relevant repair information found."
    
    formatted = "\n" + "="*60 + "\n"
    formatted += "REPAIR SEARCH RESULTS\n"
    formatted += "="*60 + "\n"
    
    for i, result in enumerate(results, 1):
        formatted += f"\n[{i}] {result['device']} Repair (Relevance: {result['relevance_score']:.2f})\n"
        formatted += "-" * 40 + "\n"
        formatted += f"{result['content']}\n"
        formatted += "-" * 40 + "\n"
    
    return formatted

# Interactive search function
def interactive_search():
    """Run interactive search session"""
    print("🔧 Repair Assistant - Ask me about Mac or iPhone repairs!")
    print("Type 'quit' to exit\n")
    
    while True:
        query = input("What repair issue can I help you with? ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        
        if not query:
            continue
        
        # Determine device type from query (basic keyword detection)
        device_type = "both"
        if "mac" in query.lower() or "imac" in query.lower() or "macbook" in query.lower():
            device_type = "mac"
        elif "iphone" in query.lower():
            device_type = "iphone"
        
        print(f"\nSearching for: '{query}' (Device: {device_type})")
        print("🔍 Searching...")
        
        results = search_repairs(query, device_type=device_type, top_k=3)
        print(format_results(results))
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    # Test the search functionality
    print("Testing search functionality...")
    
    # Example searches
    test_queries = [
        "screen replacement",
        "battery issues",
        "won't turn on",
        "water damage"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        results = search_repairs(query, top_k=2)
        print(format_results(results))
    
    # Start interactive mode
    print("\nStarting interactive mode...")
    interactive_search()
