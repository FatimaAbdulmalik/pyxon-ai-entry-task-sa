import sqlite3

DB_NAME = "pyxon_enterprise.db"

def init_db():
    """إنشاء قاعدة البيانات والداول إذا لم تكن موجودة"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 1. جدول المستندات (لحفظ البيانات الوصفية الكبيرة للتحجيم Scalability)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            file_name TEXT,
            file_type TEXT,
            chunk_strategy TEXT,
            total_chunks INTEGER,
            parsing_time REAL
        )
    ''')
    
    # 2. جدول الـ Chunks تفصيلياً (يرتبط بالجدول الأول)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            doc_id TEXT,
            text TEXT,
            page_number INTEGER,
            token_count INTEGER,
            FOREIGN KEY(doc_id) REFERENCES documents(id)
        )
    ''')
    conn.commit()
    conn.close()

def save_document_to_sql(doc_id, name, f_type, strategy, total_chunks, p_time, chunks_list):
    """حفظ سجلات المستند والقطع داخل الـ SQL"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # حفظ في جدول المستندات الرئيسي
    cursor.execute("INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?)", 
                   (doc_id, name, f_type, strategy, total_chunks, p_time))
    
    # حفظ كل قطعة على حدة مع الـ Metadata حقتها
    for ch in chunks_list:
        cursor.execute("INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)",
                       (ch['chunk_id'], doc_id, ch['text'], ch['page'], ch['tokens']))
    conn.commit()
    conn.close()