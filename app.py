import streamlit as st
import os
import time
import uuid
import pandas as pd
import sqlite3
from docx import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.docstore.document import Document as LCDoc

import database
import core
import benchmark

database.init_db()

st.set_page_config(page_title="Pyxon AI Enterprise RAG Suite", layout="wide")
st.title("🛡️ Enterprise AI Document Parser & Benchmark Suite")
st.caption("نظام هندسي متكامل يطبق تقنيات الـ Adaptive Chunking والتخزين الثنائي مع حزمة قياس الأداء")

uploaded_file = st.file_uploader("ارفع ملفك الآن (PDF, DOCX, TXT):", type=["pdf", "docx", "txt"])

if uploaded_file:
    file_id = str(uuid.uuid4())[:8]
    file_name = uploaded_file.name
    file_ext = os.path.splitext(file_name)[1].lower()
    
    start_parse = time.time()
    raw_text = ""
    
    if file_ext == ".pdf":
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader("temp.pdf")
        raw_text = " ".join([page.page_content for page in loader.load()])
    elif file_ext == ".docx":
        doc = Document(uploaded_file)
        raw_text = " ".join([p.text for p in doc.paragraphs])
    elif file_ext == ".txt":
        raw_text = uploaded_file.read().decode("utf-8")
        
    clean_text = core.normalize_arabic(raw_text)
    chunks, strategy_used = core.adaptive_chunking(clean_text)
    end_parse = time.time()
    parse_duration = round(end_parse - start_parse, 4)
    
    structured_chunks = []
    lc_docs = []
    for i, ch_text in enumerate(chunks):
        metadata = {
            "file_name": file_name,
            "page": i + 1,
            "chunk_id": f"{file_id}_ch_{i}",
            "language": "ar",
            "tokens": len(ch_text.split()),
            "chunk_strategy": strategy_used
        }
        structured_chunks.append({
            "chunk_id": metadata["chunk_id"],
            "text": ch_text,
            "page": metadata["page"],
            "tokens": metadata["tokens"]
        })
        lc_docs.append(LCDoc(page_content=ch_text, metadata=metadata))
        
    database.save_document_to_sql(file_id, file_name, file_ext, strategy_used, len(chunks), parse_duration, structured_chunks)
    
    with st.spinner("جاري تكشيف البيانات وحفظها مستمراً في ChromaDB باستخدام موديل BGE-M3..."):
        emb_model = core.get_embedding_model()
vector_store = Chroma.from_documents(documents=lc_docs, embedding=emb_model, persist_directory="./chroma_enterprise_db_new")

    st.success("✅ تم تفعيل معايير الإنتاجية بنجاح!")
    
    st.subheader("📊 لوحة النظام الهندسية (System Insights & Metrics)")
    col1, col2, col3 = st.columns(3)
    col1.metric("إستراتيجية التقطيع التكيفية المستخدمة", strategy_used)
    col2.metric("عدد الـ Chunks المستخرجة وموقعها", f"{len(chunks)} قطعة محفوظة")
    col3.metric("وقت معالجة وتكشيف النص الكلي", f"{parse_duration} ثانية")
    
    with st.expander("👁️ سجلات قاعدة بيانات SQL Metadata الحية (Enterprise Logs)"):
        import pandas as pd
        conn = sqlite3.connect(database.DB_NAME)
        df_docs = pd.read_sql_query("SELECT * FROM documents ORDER BY id DESC", conn)
        st.dataframe(df_docs, use_container_width=True)
        conn.close()

    st.subheader("🔍 محرك الاسترجاع واختبار الأداء")
    query = st.text_input("اسأل أي سؤال لاختبار دقة وسرعة النظام الحقيقي:")
    
    if query:
        initial_results = vector_store.similarity_search(query, k=5)
        final_results = core.simple_reranker(query, initial_results)[:3]
        bench_res = benchmark.run_retrieval_benchmark(vector_store, query)
        
        st.markdown(f"**⏱️ حزمة قياس الأداء الحالية (Benchmark Metrics):** Latency: {bench_res['latency_sec']}s | Accuracy: {bench_res['retrieval_accuracy']} | `Recall@K: {bench_res['recall_at_k']}`")
        
        st.markdown("### 💡 أقرب النتائج المسترجعة بدقة بعد الـ Rerank:")
        for idx, doc in enumerate(final_results):
            st.info(f"النتيجة {idx+1} | المعرف الذكي للقطعة: {doc.metadata.get('chunk_id')} | الصفحة: `{doc.metadata.get('page')}`")
            st.write(doc.page_content)
