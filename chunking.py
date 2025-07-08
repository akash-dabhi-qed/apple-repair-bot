import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.tokenize import sent_tokenize
import pandas as pd
from tqdm import tqdm
import nltk
from transformers import AutoTokenizer

# Download NLTK data for sentence tokenization
nltk.download('punkt')

class RepairManualChunker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        # Configure splitter for repair manuals
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # Optimal for all-mpnet-base-v2
            chunk_overlap=50,  # Maintain context between chunks
            length_function=self.token_length,
            separators=["\n\n## ", "\n## ", "\n\n", "\n", " "]  # Markdown-friendly
        )
        
        # Repair-specific patterns
        self.warning_pattern = re.compile(r"WARNING:(.*?)(?:\n\n|\Z)", re.DOTALL)
        self.caution_pattern = re.compile(r"CAUTION:(.*?)(?:\n\n|\Z)", re.DOTALL)
    
    def preserve_safety_notices(self, text):
        # Extract safety warnings and preserve with related content
        warnings = self.warning_pattern.findall(text)
        cautions = self.caution_pattern.findall(text)
        return warnings + cautions
    
    def preprocess_manual(self, text):
        # Normalize text
        text = text.replace('\r\n', '\n').replace('\t', ' ')
        
        # Preserve section headers
        text = re.sub(r'\n(\d+\.\d+)', r'\n## \1', text)  # Numbered sections
        return text
    
    def token_length(self, text):
        return len(self.tokenizer.encode(text))
    
    def chunk_manual(self, text, manual_id):
        # Preprocess
        processed_text = self.preprocess_manual(text)
        
        # Preserve safety notices
        safety_notices = self.preserve_safety_notices(processed_text)
        
        # Split into chunks
        chunks = self.splitter.split_text(processed_text)
        
        # Add safety notices to relevant chunks
        for i, chunk in enumerate(chunks):
            for notice in safety_notices:
                if notice.lower() in chunk.lower():
                    chunks[i] = f"SAFETY: {notice}\n\n{chunk}"
        
        # Prepare output with metadata
        chunk_records = []
        for i, chunk in enumerate(chunks):
            record = {
                "chunk_id": f"{manual_id}_{i}",
                "text": chunk,
                "manual_id": manual_id,
                "contains_warning": any(notice.lower() in chunk.lower() for notice in safety_notices),
                "chunk_length": self.token_length(chunk)
            }
            chunk_records.append(record)
            
        return chunk_records

def process_all_manuals(input_dir="data/raw", output_file="data/chunks.csv"):
    # Initialize
    chunker = RepairManualChunker()
    all_chunks = []
    
    # Process each file
    manual_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    for filename in tqdm(manual_files, desc="Processing manuals"):
        manual_id = os.path.splitext(filename)[0]
        with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunks = chunker.chunk_manual(text, manual_id)
        all_chunks.extend(chunks)
    
    # Save to CSV
    df = pd.DataFrame(all_chunks)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} chunks to {output_file}")

if __name__ == "__main__":
    # Create data directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    print("Repair Manual Chunking System")
    print("1. Place your repair manuals in data/raw/ as .txt files")
    print("2. Processing will begin automatically...")
    
    process_all_manuals()
