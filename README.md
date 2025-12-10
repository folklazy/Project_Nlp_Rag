# Book Chat RAG: The Heart of a Woman

โครงการนี้คือแอปพลิเคชัน Chatbot บน Streamlit ที่สร้างขึ้นเพื่อใช้เทคนิค **Retrieval-Augmented Generation (RAG)** ในการตอบคำถามเกี่ยวกับเนื้อหาในหนังสือ "The Heart of a Woman" (โดย Baroness Orczy) โดยอิงข้อมูลจากฐานข้อมูลเวกเตอร์ **ChromaDB** ที่ถูกสร้างไว้ล่วงหน้า

---

###  เทคโนโลยีหลักที่ใช้

* **Streamlit:** สำหรับการสร้างส่วนต่อประสานกับผู้ใช้ (User Interface) แบบ Interactive
* **ChromaDB:** ฐานข้อมูลเวกเตอร์ (Vector Database) ที่ใช้เก็บข้อมูลแบบฝังตัว (Embeddings)
* **SentenceTransformer (all-mpnet-base-v2):** โมเดลสำหรับสร้าง Embeddings (Retriever)
* **Mistral-7B-Instruct-v0.2:** โมเดลภาษาขนาดใหญ่ (LLM) ที่ใช้ในการสร้างคำตอบจากบริบทที่ดึงมา (Generator)
* **Hugging Face Transformers / PyTorch:** สำหรับจัดการและรันโมเดล LLM

---

###  การเริ่มต้นใช้งาน (Getting Started)

#### 1. ข้อกำหนดเบื้องต้น (Prerequisites)

* Python 3.x
* **ฐานข้อมูล ChromaDB:** โครงการนี้ต้องการโฟลเดอร์ฐานข้อมูล ChromaDB ชื่อ `./chroma_db_optimized` ที่มี Collection ชื่อ `baroness_orczy_optimized` อยู่ภายใน **ก่อน** ที่จะรันแอปพลิเคชัน
* การ์ดจอที่รองรับ CUDA (แนะนำอย่างยิ่ง) สำหรับการรัน LLM

#### 2. การติดตั้ง (Installation)

ติดตั้งไลบรารีที่จำเป็นทั้งหมดผ่าน pip:

```bash
pip install streamlit chromadb sentence-transformers transformers torch
```

#### 3. การล้างแคช (ถ้าจำเป็น)

หากมีการแก้ไขโค้ดส่วนการโหลดทรัพยากรที่ใช้ @st.cache_resource หรือเปลี่ยนโมเดล คุณอาจจำเป็นต้องล้างแคช Streamlit ก่อนรัน:

```bash
python -m streamlit cache clear
```

#### 4. รันแอปพลิเคชัน

เริ่มต้นแอปพลิเคชัน Streamlit:

```bash
python -m streamlit run app.py
```

---

### หลักการทำงานของ RAG
แอปพลิเคชันนี้ใช้สถาปัตยกรรม RAG เพื่อให้ LLM สามารถตอบคำถามที่เฉพาะเจาะจงกับเนื้อหาหนังสือได้ โดยมีขั้นตอนดังนี้:

* **การฝังตัวคำถาม (Query Embedding):** คำถามของผู้ใช้จะถูกแปลงเป็นเวกเตอร์โดย SentenceTransformer

* **การค้นคืน (Retrieval):** เวกเตอร์คำถามจะใช้ในการค้นคืนข้อความที่ "คล้ายกัน" มากที่สุด 5 ส่วนจาก ChromaDB

* **การสร้างบริบท (Context Formation):** ข้อความที่ค้นคืนมาจะถูกรวมเข้าด้วยกันเพื่อสร้างบริบทที่สมบูรณ์

* **การสร้างคำตอบ (Generation):** บริบทและคำถามจะถูกส่งไปยัง Mistral-7B เพื่อให้สร้างคำตอบที่อิงจากข้อมูลในบริบทเท่านั้น

---

### การจัดการข้อผิดพลาด (Error Handling)
* **LLM ไม่พร้อม:** หากการโหลดโมเดล Mistral ล้มเหลว (เช่น หน่วยความจำไม่พอ) แอปพลิเคชันจะยังทำงานได้ แต่จะแจ้งเตือน 
**⚠️ LLM โหลดไม่ได้** และไม่สามารถสร้างคำตอบได้

* **ChromaDB ไม่พร้อม:** หากไม่พบโฟลเดอร์ ./chroma_db_optimized หรือ Collection ที่ต้องการ แอปพลิเคชันจะแจ้งข้อผิดพลาดและไม่สามารถทำ RAG ได้เลย