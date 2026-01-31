# =============================================================================
# IMPROVED CHUNKING SCRIPT FOR RAG
# =============================================================================
# ปรับปรุงการ chunk ให้มี overlap และขนาดสม่ำเสมอ
# เหมาะสำหรับ all-mpnet-base-v2 embedding model (max 384 tokens)
# =============================================================================

import os
import re
import math
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

# =============================================================================
# CONFIGURATION
# =============================================================================

# Chunking Parameters
CHUNK_SIZE = 500        # จำนวน characters ต่อ chunk (เหมาะสมกับ embedding model)
CHUNK_OVERLAP = 100     # overlap ระหว่าง chunks (รักษา context)
MIN_CHUNK_SIZE = 50     # ขนาดขั้นต่ำของ chunk

# Paths
DATASET_PATH = "./notebook/datasets"
FILE_NAME = "Baronness Orczy___The Heart of a Woman.txt"
CHROMA_PATH = "./chroma_db_optimized"
COLLECTION_NAME = "baroness_orczy_optimized"

# =============================================================================
# RECURSIVE CHARACTER TEXT SPLITTER
# =============================================================================

class RecursiveCharacterTextSplitter:
    """
    Senior-Level Text Splitter ที่แบ่งข้อความอย่างฉลาด
    - พยายามแบ่งตาม paragraph > sentence > word
    - มี overlap เพื่อรักษา context
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: list = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(self, text: str) -> list:
        """แบ่ง text เป็น chunks"""
        return self._split_text_recursive(text, self.separators)
    
    def _split_text_recursive(self, text: str, separators: list) -> list:
        """แบ่ง text โดยใช้ separator ที่เหมาะสมที่สุด"""
        final_chunks = []
        
        # หา separator ที่ใช้ได้
        separator = separators[-1]
        new_separators = []
        
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break
        
        # แบ่ง text ด้วย separator
        splits = text.split(separator) if separator else list(text)
        
        # รวม splits ให้ได้ขนาดที่ต้องการ
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split)
            
            # ถ้าเพิ่ม split แล้วยังไม่เกิน chunk_size
            if current_length + split_length + len(separator) <= self.chunk_size:
                current_chunk.append(split)
                current_length += split_length + len(separator)
            else:
                # บันทึก chunk ปัจจุบัน
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    
                    # ถ้า chunk ยังใหญ่เกินไป ให้แบ่งต่อ
                    if len(chunk_text) > self.chunk_size and new_separators:
                        final_chunks.extend(
                            self._split_text_recursive(chunk_text, new_separators)
                        )
                    else:
                        final_chunks.append(chunk_text)
                
                # เริ่ม chunk ใหม่พร้อม overlap
                overlap_text = self._get_overlap_text(current_chunk, separator)
                current_chunk = [overlap_text, split] if overlap_text else [split]
                current_length = len(separator.join(current_chunk))
        
        # บันทึก chunk สุดท้าย
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if len(chunk_text) > self.chunk_size and new_separators:
                final_chunks.extend(
                    self._split_text_recursive(chunk_text, new_separators)
                )
            else:
                final_chunks.append(chunk_text)
        
        return final_chunks
    
    def _get_overlap_text(self, chunks: list, separator: str) -> str:
        """ดึง overlap text จาก chunks ก่อนหน้า"""
        if not chunks or self.chunk_overlap <= 0:
            return ""
        
        full_text = separator.join(chunks)
        if len(full_text) <= self.chunk_overlap:
            return full_text
        
        return full_text[-self.chunk_overlap:]


# =============================================================================
# TEXT CLEANING
# =============================================================================

def clean_gutenberg_text(text: str) -> str:
    """ทำความสะอาด text จาก Project Gutenberg"""
    text = re.sub(r'[_*#=~"\u201c\u201d\'\u2014]+', ' ', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_book(text: str) -> str:
    """ทำความสะอาดและเตรียม text ก่อน chunking"""
    # ลบ Gutenberg header/footer
    lines = text.split('\n')
    clean_lines = []
    in_content = False
    
    for line in lines:
        line_upper = line.upper().strip()
        
        # ข้าม Gutenberg metadata
        if "PROJECT GUTENBERG" in line_upper or "*** START" in line_upper:
            in_content = True
            continue
        if "*** END" in line_upper:
            break
        
        if in_content or not line_upper.startswith("***"):
            clean_lines.append(line)
    
    text = '\n'.join(clean_lines)
    text = clean_gutenberg_text(text)
    return text


# =============================================================================
# MAIN CHUNKING FUNCTION
# =============================================================================

def create_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chunk_size: int = MIN_CHUNK_SIZE
) -> list:
    """สร้าง chunks จาก text"""
    
    # สร้าง splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # แบ่ง chunks
    chunks = splitter.split_text(text)
    
    # กรอง chunks ที่สั้นเกินไป
    filtered_chunks = [
        chunk for chunk in chunks 
        if len(chunk) >= min_chunk_size
    ]
    
    return filtered_chunks


# =============================================================================
# BUILD VECTOR DATABASE
# =============================================================================

def build_vector_db(chunks: list, chroma_path: str, collection_name: str):
    """สร้าง ChromaDB จาก chunks"""
    
    # โหลด embedding model
    print("Loading embedding model...")
    model = SentenceTransformer("all-mpnet-base-v2")
    print(f"✅ Loaded: all-mpnet-base-v2 (768 dimensions)")
    
    # สร้าง ChromaDB client
    client = chromadb.PersistentClient(path=chroma_path)
    
    # ลบ collection เดิม (ถ้ามี)
    try:
        client.delete_collection(name=collection_name)
        print(f"🗑️ Deleted existing collection: {collection_name}")
    except:
        pass
    
    # สร้าง collection ใหม่
    collection = client.get_or_create_collection(name=collection_name)
    
    # Upload chunks แบบ batch
    batch_size = 256
    num_batches = math.ceil(len(chunks) / batch_size)
    
    print(f"\n📤 Uploading {len(chunks)} chunks in {num_batches} batches...")
    
    for i in tqdm(range(num_batches), desc="Uploading"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(chunks))
        
        batch_chunks = chunks[start_idx:end_idx]
        batch_embeddings = model.encode(batch_chunks, show_progress_bar=False).tolist()
        batch_ids = [f"chunk_{start_idx + j}" for j in range(len(batch_chunks))]
        
        collection.add(
            documents=batch_chunks,
            embeddings=batch_embeddings,
            ids=batch_ids
        )
    
    print(f"\n✅ Done! Total documents: {collection.count()}")
    return collection


# =============================================================================
# COMPARISON FUNCTION
# =============================================================================

def compare_chunking_methods(text: str):
    """เปรียบเทียบ chunking แบบเก่าและใหม่"""
    
    print("=" * 60)
    print("📊 CHUNKING COMPARISON")
    print("=" * 60)
    
    # วิธีเก่า (Paragraph-based)
    paragraphs = text.split('\n\n')
    old_chunks = [
        clean_gutenberg_text(p) for p in paragraphs 
        if len(p.split()) >= 10 and not p.upper().startswith("CHAPTER")
    ]
    old_chunks = [c for c in old_chunks if c]
    
    # วิธีใหม่ (Recursive with overlap)
    new_chunks = create_chunks(text)
    
    # แสดงผลเปรียบเทียบ
    print(f"\n{'Metric':<30} {'Old Method':<15} {'New Method':<15}")
    print("-" * 60)
    print(f"{'Number of chunks':<30} {len(old_chunks):<15} {len(new_chunks):<15}")
    
    old_avg = sum(len(c) for c in old_chunks) / len(old_chunks) if old_chunks else 0
    new_avg = sum(len(c) for c in new_chunks) / len(new_chunks) if new_chunks else 0
    print(f"{'Average chunk size (chars)':<30} {old_avg:<15.0f} {new_avg:<15.0f}")
    
    old_min = min(len(c) for c in old_chunks) if old_chunks else 0
    new_min = min(len(c) for c in new_chunks) if new_chunks else 0
    print(f"{'Min chunk size':<30} {old_min:<15} {new_min:<15}")
    
    old_max = max(len(c) for c in old_chunks) if old_chunks else 0
    new_max = max(len(c) for c in new_chunks) if new_chunks else 0
    print(f"{'Max chunk size':<30} {old_max:<15} {new_max:<15}")
    
    print(f"{'Has overlap?':<30} {'No':<15} {'Yes (100 chars)':<15}")
    
    print("\n" + "=" * 60)
    print("📝 SAMPLE CHUNKS (New Method)")
    print("=" * 60)
    for i, chunk in enumerate(new_chunks[:3]):
        print(f"\n[Chunk {i+1}] ({len(chunk)} chars)")
        print("-" * 40)
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    
    return old_chunks, new_chunks


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # อ่านไฟล์
    file_path = os.path.join(DATASET_PATH, FILE_NAME)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        exit(1)
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()
    
    print(f"📖 Loaded: {FILE_NAME}")
    print(f"   Raw text length: {len(raw_text):,} characters")
    
    # Preprocess
    text = preprocess_book(raw_text)
    print(f"   Cleaned text length: {len(text):,} characters")
    
    # เปรียบเทียบ
    old_chunks, new_chunks = compare_chunking_methods(text)
    
    # ถามว่าจะ build database ไหม
    print("\n" + "=" * 60)
    user_input = input("🔄 Build new ChromaDB with improved chunks? (y/n): ")
    
    if user_input.lower() == 'y':
        build_vector_db(new_chunks, CHROMA_PATH, COLLECTION_NAME)
        print("\n🎉 Vector database rebuilt successfully!")
    else:
        print("\n⏭️ Skipped database rebuild.")
