import streamlit as st
import os
import chromadb
from sentence_transformers import SentenceTransformer
import torch

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    RAG_LLM_READY = True
except Exception:
    RAG_LLM_READY = False
    st.warning("⚠️ Transformers หรือ Torch ยังไม่พร้อม — ระบบจะทำงานแบบไม่มี LLM")

@st.cache_resource
def load_rag_components():
    embed_model = None
    db_collection = None
    rag_pipeline = None

    try:
        embed_model = SentenceTransformer("all-mpnet-base-v2")
    except Exception as e:
        st.error(f"❌ โหลด SentenceTransformer ไม่สำเร็จ: {e}")
        return None, None, None

    CHROMA_PATH = "./chroma_db_optimized"
    COLLECTION_NAME = "baroness_orczy_optimized"

    try:
        if not os.path.isdir(CHROMA_PATH):
            st.error(f"❌ ไม่พบโฟลเดอร์ ChromaDB: {CHROMA_PATH}")
            return embed_model, None, None
        
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        db_collection = client.get_collection(name=COLLECTION_NAME)

    except Exception as e:
        st.error(f"❌ โหลด ChromaDB ไม่สำเร็จ: {e}")
        return embed_model, None, None

    if not RAG_LLM_READY:
        return embed_model, db_collection, None

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    try:
        with st.spinner(f"กำลังโหลด LLM {model_name} ..."):

            tokenizer = AutoTokenizer.from_pretrained(model_name)

            llm_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )

            rag_pipeline = pipeline(
                "text2text-generation",
                model=llm_model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )

    except Exception as e:
        st.error(f"❌ โหลด LLM ไม่สำเร็จ: {e}")
        return embed_model, db_collection, None

    return embed_model, db_collection, rag_pipeline

embed_model, db_collection, rag_pipeline = load_rag_components()

def get_rag_answer(query_text, embed_model, db_collection, rag_pipeline):
    if not embed_model or not db_collection:
        return "⚠️ ไม่สามารถทำ RAG ได้: Embedding/ChromaDB ยังไม่พร้อม"

    if rag_pipeline is None:
        return "⚠️ Embedding ใช้ได้ แต่ LLM โหลดไม่สำเร็จ"

    try:
        query_embedding = embed_model.encode([query_text])
        results = db_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=5
        )
        context = "\n".join(results["documents"][0])
    except Exception as e:
        return f"❌ เกิดปัญหาในการค้นหา ChromaDB: {e}"

    try:
        prompt = f"""
Use the context below to answer the question clearly and concisely in English.
If the information is not available in the context, state that politely.

Context:
{context}

Question: {query_text}
Answer:
"""
        outputs = rag_pipeline(
            prompt,
            max_new_tokens=350,
            temperature=0.7,
            do_sample=True,
        )
        return outputs[0]['generated_text'].strip()

    except Exception as e:
        return f"❌ เกิดข้อผิดพลาดในการสร้างคำตอบจาก LLM: {e}"

st.set_page_config(page_title="Book Chat", page_icon="📚")

if "messages" not in st.session_state:
    msg = "สวัสดี! คุณสามารถถามคำถามเกี่ยวกับเนื้อหาในหนังสือ The Heart of a Woman ได้เลย 😊"
    if rag_pipeline is None:
        msg += "\n\n⚠️ LLM โหลดไม่ได้ แต่สามารถค้นหาจาก ChromaDB ได้"
    else:
        msg += "\n\nระบบพร้อมใช้งานแล้ว!"
    st.session_state.messages = [("assistant", msg)]

def inject_css():
    st.markdown("""
<style>
.block-container {padding-top: 80px !important;}
.fixed-header {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 60px;
    background-color: white;
    border-bottom: 1px solid #e0e0e0;
    display: flex;
    align-items: center;
    justify-content: left;
    padding: 0 20px;
    z-index: 9999;
}
.header-title {font-weight: 700; font-size: 22px;}
</style>
""", unsafe_allow_html=True)


def chat_page():
    inject_css()
    st.markdown('<div class="fixed-header"><span class="header-title">📚 Book Chat RAG</span></div>', unsafe_allow_html=True)

    for role, msg in st.session_state.messages:
        with st.chat_message(role):
            st.write(msg)

    prompt = st.chat_input("พิมพ์คำถาม...")
    if prompt:
        st.session_state.messages.append(("user", prompt))
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหาและสร้างคำตอบ..."):
                reply = get_rag_answer(prompt, embed_model, db_collection, rag_pipeline)
            st.write(reply)

        st.session_state.messages.append(("assistant", reply))

chat_page()
