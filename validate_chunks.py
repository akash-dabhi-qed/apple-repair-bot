import pandas as pd
import numpy as np
import re
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class ChunkingValidator:
    def __init__(self):
        self.repair_types = {
            'screen': ['screen', 'display', 'lcd', 'digitizer', 'touch'],
            'battery': ['battery', 'power', 'charging', 'charge'],
            'camera': ['camera', 'lens', 'photo', 'flash'],
            'keyboard': ['keyboard', 'key', 'trackpad', 'touchpad'],
            'logic_board': ['logic board', 'motherboard', 'cpu', 'gpu'],
            'speaker': ['speaker', 'audio', 'sound', 'microphone'],
            'port': ['port', 'charging port', 'usb', 'lightning'],
            'case': ['case', 'housing', 'back cover', 'frame']
        }
        
        self.safety_keywords = [
            'warning', 'caution', 'danger', 'careful', 'safety',
            'hot', 'sharp', 'electric', 'disconnect', 'power off'
        ]
    
    def load_and_basic_check(self, file_path, device_type):
        """Load CSV and perform basic structural checks"""
        print(f"\n{'='*50}")
        print(f"VALIDATING {device_type.upper()} CHUNKS")
        print(f"{'='*50}")
        
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Successfully loaded {file_path}")
            print(f"📊 Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            return df
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return None
    
    def check_columns(self, df, device_type):
        """Check column structure and required fields"""
        print(f"\n📋 COLUMN ANALYSIS")
        print("-" * 30)
        
        required_cols = ['chunk_id', 'text']
        recommended_cols = ['manual_id', 'chunk_length', 'contains_warning']
        
        actual_cols = df.columns.tolist()
        print(f"Actual columns: {actual_cols}")
        
        # Check required columns
        missing_required = [col for col in required_cols if col not in actual_cols]
        if missing_required:
            print(f"❌ Missing required columns: {missing_required}")
        else:
            print(f"✅ All required columns present")
        
        # Check recommended columns
        missing_recommended = [col for col in recommended_cols if col not in actual_cols]
        if missing_recommended:
            print(f"⚠️  Missing recommended columns: {missing_recommended}")
        else:
            print(f"✅ All recommended columns present")
        
        return actual_cols
    
    def analyze_text_content(self, df, device_type):
        """Analyze the text content quality"""
        print(f"\n📝 TEXT CONTENT ANALYSIS")
        print("-" * 30)
        
        if 'text' not in df.columns:
            print("❌ No 'text' column found")
            return
        
        # Basic text statistics
        text_lengths = df['text'].str.len()
        word_counts = df['text'].str.split().str.len()
        
        print(f"Text length stats:")
        print(f"  • Mean: {text_lengths.mean():.0f} characters")
        print(f"  • Median: {text_lengths.median():.0f} characters")
        print(f"  • Min: {text_lengths.min()} characters")
        print(f"  • Max: {text_lengths.max()} characters")
        
        print(f"\nWord count stats:")
        print(f"  • Mean: {word_counts.mean():.0f} words")
        print(f"  • Median: {word_counts.median():.0f} words")
        
        # Check for empty or very short chunks
        empty_chunks = df[df['text'].str.len() < 10].shape[0]
        if empty_chunks > 0:
            print(f"⚠️  {empty_chunks} chunks with less than 10 characters")
        
        # Check for very long chunks (might not fit in embedding model)
        long_chunks = df[df['text'].str.len() > 2000].shape[0]
        if long_chunks > 0:
            print(f"⚠️  {long_chunks} chunks longer than 2000 characters")
    
    def detect_repair_types(self, df, device_type):
        """Detect and categorize repair types from text content"""
        print(f"\n🔧 REPAIR TYPE DETECTION")
        print("-" * 30)
        
        if 'text' not in df.columns:
            return
        
        # Create repair type column if it doesn't exist
        if 'repair_type' not in df.columns:
            df['repair_type'] = 'unknown'
        
        # Detect repair types based on keywords
        for repair_type, keywords in self.repair_types.items():
            mask = df['text'].str.lower().str.contains('|'.join(keywords), na=False)
            df.loc[mask, 'repair_type'] = repair_type
        
        # Count repair types
        repair_counts = df['repair_type'].value_counts()
        print("Detected repair types:")
        for repair_type, count in repair_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  • {repair_type}: {count:,} chunks ({percentage:.1f}%)")
        
        return df
    
    def check_safety_content(self, df, device_type):
        """Check for safety warnings and important notices"""
        print(f"\n⚠️  SAFETY CONTENT ANALYSIS")
        print("-" * 30)
        
        if 'text' not in df.columns:
            return
        
        # Count safety-related content
        safety_pattern = '|'.join(self.safety_keywords)
        safety_chunks = df['text'].str.lower().str.contains(safety_pattern, na=False)
        safety_count = safety_chunks.sum()
        
        print(f"Chunks with safety content: {safety_count:,} ({(safety_count/len(df)*100):.1f}%)")
        
        # Update contains_warning column if it exists
        if 'contains_warning' not in df.columns:
            df['contains_warning'] = safety_chunks
        
        # Show sample safety content
        if safety_count > 0:
            print("\nSample safety warnings:")
            safety_samples = df[safety_chunks]['text'].head(3)
            for i, text in enumerate(safety_samples, 1):
                preview = text[:100] + "..." if len(text) > 100 else text
                print(f"  {i}. {preview}")
        
        return df
    
    def check_data_quality(self, df, device_type):
        """Check for data quality issues"""
        print(f"\n🔍 DATA QUALITY CHECKS")
        print("-" * 30)
        
        # Check for duplicates
        if 'text' in df.columns:
            duplicates = df['text'].duplicated().sum()
            print(f"Duplicate chunks: {duplicates}")
            if duplicates > 0:
                print("⚠️  Consider removing duplicate chunks")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("Missing values:")
            for col, count in missing_values.items():
                if count > 0:
                    print(f"  • {col}: {count}")
        
        # Check chunk ID format
        if 'chunk_id' in df.columns:
            unique_ids = df['chunk_id'].nunique()
            total_rows = len(df)
            if unique_ids != total_rows:
                print(f"⚠️  Non-unique chunk IDs: {unique_ids} unique vs {total_rows} total")
    
    def create_sample_dataset(self, df, device_type, sample_size=100):
        """Create a balanced sample for testing"""
        print(f"\n📊 CREATING SAMPLE DATASET")
        print("-" * 30)
        
        if 'repair_type' not in df.columns:
            # Random sample if no repair types
            sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        else:
            # Stratified sample by repair type
            samples_per_type = sample_size // df['repair_type'].nunique()
            sample_dfs = []
            
            for repair_type in df['repair_type'].unique():
                type_df = df[df['repair_type'] == repair_type]
                n_samples = min(samples_per_type, len(type_df))
                if n_samples > 0:
                    sample_dfs.append(type_df.sample(n=n_samples, random_state=42))
            
            sample_df = pd.concat(sample_dfs, ignore_index=True)
        
        sample_file = f"sample_{device_type}_chunks.csv"
        sample_df.to_csv(sample_file, index=False)
        print(f"✅ Created sample dataset: {sample_file} ({len(sample_df)} chunks)")
        
        return sample_df
    
    def generate_report(self, df, device_type):
        """Generate summary report"""
        print(f"\n📋 SUMMARY REPORT - {device_type.upper()}")
        print("=" * 50)
        
        print(f"Dataset size: {len(df):,} chunks")
        
        if 'repair_type' in df.columns:
            print(f"Repair categories: {df['repair_type'].nunique()}")
        
        if 'contains_warning' in df.columns:
            safety_pct = (df['contains_warning'].sum() / len(df)) * 100
            print(f"Safety content: {safety_pct:.1f}%")
        
        if 'text' in df.columns:
            avg_length = df['text'].str.len().mean()
            print(f"Average chunk length: {avg_length:.0f} characters")
        
        # Recommendations
        print(f"\n📝 RECOMMENDATIONS:")
        recommendations = []
        
        if len(df) > 10000:
            recommendations.append("• Consider sampling for initial development")
        
        if 'repair_type' not in df.columns:
            recommendations.append("• Add repair_type categorization")
        
        if 'contains_warning' not in df.columns:
            recommendations.append("• Add safety warning detection")
        
        if df['text'].str.len().max() > 2000:
            recommendations.append("• Review very long chunks for embedding compatibility")
        
        for rec in recommendations:
            print(rec)

def main():
    validator = ChunkingValidator()
    
    mac_file = "./mac_chunks.csv" 
    iphone_file = "./iphone_chunks.csv" 
    
    print("🔍 CHUNKING CSV VALIDATION TOOL")
    print("=" * 50)
    
    # Validate Mac chunks
    if os.path.exists(mac_file):
        mac_df = validator.load_and_basic_check(mac_file, "Mac")
        if mac_df is not None:
            validator.check_columns(mac_df, "Mac")
            validator.analyze_text_content(mac_df, "Mac")
            mac_df = validator.detect_repair_types(mac_df, "Mac")
            mac_df = validator.check_safety_content(mac_df, "Mac")
            validator.check_data_quality(mac_df, "Mac")
            validator.create_sample_dataset(mac_df, "mac", sample_size=100)
            validator.generate_report(mac_df, "Mac")
    else:
        print(f"⚠️  Mac file not found: {mac_file}")
    
    # Validate iPhone chunks
    if os.path.exists(iphone_file):
        iphone_df = validator.load_and_basic_check(iphone_file, "iPhone")
        if iphone_df is not None:
            validator.check_columns(iphone_df, "iPhone")
            validator.analyze_text_content(iphone_df, "iPhone")
            iphone_df = validator.detect_repair_types(iphone_df, "iPhone")
            iphone_df = validator.check_safety_content(iphone_df, "iPhone")
            validator.check_data_quality(iphone_df, "iPhone")
            validator.create_sample_dataset(iphone_df, "iphone", sample_size=100)
            validator.generate_report(iphone_df, "iPhone")
    else:
        print(f"⚠️  iPhone file not found: {iphone_file}")
    
    print(f"\n✅ VALIDATION COMPLETE!")
    print("Check for sample_mac_chunks.csv and sample_iphone_chunks.csv files")

if __name__ == "__main__":
    main()
