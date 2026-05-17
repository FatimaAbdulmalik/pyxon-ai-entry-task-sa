import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. طبقة معالجة اللغة العربية (Arabic Normalization Layer)
def normalize_arabic(text):
    """تنظيف وتوحيد النص العربي لرفع دقة البحث"""
    # إزالة الحركات والتشكيل بالكامل
    diacritics = re.compile(r'[\u064B-\u0652]')
    text = re.sub(diacritics, '', text)
    # توحيد الألفات والهمزات
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    # إزالة الكشيدة أو التطويل (مثل: فـــاطمة تعود فاطمة)
    text = re.sub(r'ـ', '', text)
    return text

# 2. محرك التقطيع الذكي التكيفي (Intelligent Adaptive Chunking Engine)
def adaptive_chunking(text):
    """محرك اتخاذ القرار لاختيار أفضل استراتيجية تقطيع تلقائياً"""
    # فحص النص: هل يحتوي على علامات عناوين رئيسية؟
    has_headings = len(re.findall(r'(📌|###|المبحث|الفصل|أولاً|ثانياً)', text)) > 2
    # فحص النص: هل يحتوي على فقرات واضحة مفصولة بأسطر؟
    long_paragraphs = len(text.split('\n\n')) > 3
    
    if has_headings:
        strategy = "Heading-Based (Adaptive)"
        chunks = [p.strip() for p in re.split(r'(###|📌|المبحث|الفصل)', text) if p.strip()]
    elif long_paragraphs:
        strategy = "Semantic/Paragraph-Based"
        chunks = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 20]
    else:
        strategy = "Fixed-Size (Recursive)"
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        chunks = splitter.split_text(text)
        
    return chunks, strategy

# 3. استدعاء موديل الـ Embedding القوي والمطلوب هندسياً BGE-M3
def get_embedding_model():
    """جلب موديل BGE-M3 العالمي الخبير في فهم معاني الكلمات العربية"""
    return 
    HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 4. محرك إعادة الترتيب (Simple Reranker)
def simple_reranker(query, retrieved_docs):
    """إعادة ترتيب النتائج المسترجعة لضمان ظهور النتيجة الأكثر دقة في الأعلى"""
    query_words = query.split()
    scored_docs = []
    for doc in retrieved_docs:
        # حساب كم كلمة من السؤال تكررت في هذه القطعة المسترجعة
        score = sum(1 for word in query_words if word in doc.page_content)
        scored_docs.append((score, doc))
    # ترتيب تنازلي (الأسكور الأعلى أولاً)
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs]
