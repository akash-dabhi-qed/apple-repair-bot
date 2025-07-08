from retriever import RepairManualRetriever

retriever = RepairManualRetriever()

queries = [
    "How to replace iPhone battery safely?",
    "My laptop won't turn on after screen replacement",
    "Washing machine leaking water repair"
]

for query in queries:
    print(f"\nQuery: {query}")
    results = retriever.search(query)
    for i, result in enumerate(results[:2]):  # Show top 2 results
        print(f"\nResult {i+1}:")
        print(f"Manual: {result['manual_id']}")
        print(f"Contains warning: {result['contains_warning']}")
        print(f"Text preview: {result['text'][:200]}...")
        print(f"Distance score: {result['distance']:.4f}")
