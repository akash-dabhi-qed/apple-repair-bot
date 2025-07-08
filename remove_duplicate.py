# Run this to clean your datasets
import pandas as pd

# Clean Mac chunks
mac_df = pd.read_csv('./mac_chunks.csv')
mac_clean = mac_df.drop_duplicates(subset=['text'], keep='first')
mac_clean.to_csv('./mac_chunks_clean.csv', index=False)
print(f"Mac: Reduced from {len(mac_df)} to {len(mac_clean)} chunks")

# Clean iPhone chunks  
iphone_df = pd.read_csv('./iphone_chunks.csv')
iphone_clean = iphone_df.drop_duplicates(subset=['text'], keep='first')
iphone_clean.to_csv('./iphone_chunks_clean.csv', index=False)
print(f"iPhone: Reduced from {len(iphone_df)} to {len(iphone_clean)} chunks")


# After deduplication, check if IDs are unique
print(f"Mac unique IDs: {mac_clean['chunk_id'].nunique()} vs Total: {len(mac_clean)}")
print(f"iPhone unique IDs: {iphone_clean['chunk_id'].nunique()} vs Total: {len(iphone_clean)}")
