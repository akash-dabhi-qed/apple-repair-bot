import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm

class DeviceWikiChunker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=self.token_length,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "]
        )
    
    def token_length(self, text):
        return len(self.tokenizer.encode(text))
    
    def chunk_wiki(self, text, wiki_id):
        chunks = self.splitter.split_text(text)
        chunk_records = []
        for i, chunk in enumerate(chunks):
            record = {
                "chunk_id": f"{wiki_id}_{i}",
                "text": chunk,
                "wiki_id": wiki_id,
                "chunk_length": self.token_length(chunk)
            }
            chunk_records.append(record)
        return chunk_records

def process_device_wikis(input_json="./../ifixit-scrape/Mac Guides/ifixit_mac_guides.json", output_file="mac_chunks.csv"):
    chunker = DeviceWikiChunker()
    all_chunks = []

    with open(input_json, "r", encoding="utf-8") as f:
        wiki_data = json.load(f)

    for doc in tqdm(wiki_data, desc="Processing Device Wikis"):
        wiki_id = doc.get("title", "unknown").replace(" ", "_")
        text = doc["content"]
        chunks = chunker.chunk_wiki(text, wiki_id)
        all_chunks.extend(chunks)

    df = pd.DataFrame(all_chunks)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} chunks to {output_file}")

if __name__ == "__main__":
    process_device_wikis()
