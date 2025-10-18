# streamlit_text_analytics_v2.py
# Веб-додаток "Текстова аналітика документів" для Streamlit
# УМОВИ: Не використовуємо PyPDF2 та pdfplumber.
# Використовуємо pdfminer.six для витягання тексту з PDF.
# Функціонал: завантаження PDF або TXT, підрахунок частот слів, побудова wordcloud та barplot,
# тематичне моделювання (LDA) через sklearn

import streamlit as st
import io
import re
from collections import Counter
from typing import List

# PDF extraction via pdfminer
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    _HAS_PDFMINER = True
except Exception:
    _HAS_PDFMINER = False

# Visualization
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    _HAS_WORDCLOUD = True
except Exception:
    _HAS_WORDCLOUD = False

# LDA via sklearn
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# Простий набір стоп-слів (англійська + українські базові)
STOPWORDS = set([
    # English (small subset)
    'the','and','is','in','it','of','to','a','that','with','as','for','its','on','be','are','this','by','an',
    # Ukrainian basic (very short list, add more as needed)
    'і','в','не','на','я','з','що','до','у','ви','ти','це','за','й','як','по','за','їх','про','між'
])

# Utility functions

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if not _HAS_PDFMINER:
        return """ERROR: Для обробки PDF потрібен пакет pdfminer.six.
Встановіть його: pip install pdfminer.six"""
    try:
        # pdfminer accepts a file path or file-like object. We'll use BytesIO
        return pdfminer_extract_text(io.BytesIO(file_bytes))
    except Exception as e:
        return f"ERROR при читанні PDF: {e}"


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_words(text: str) -> List[str]:
    words = re.findall(r"[\w\u0400-\u04FF']+", text.lower())
    return words


def get_top_n(words: List[str], n:int=20, stopwords:set=STOPWORDS):
    filtered = [w for w in words if w not in stopwords and len(w)>1]
    c = Counter(filtered)
    return c.most_common(n)


def plot_wordcloud(text: str):
    if not _HAS_WORDCLOUD:
        st.info("wordcloud не встановлено. Встановіть wordcloud для генерації хмари слів.")
        return
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)


def plot_bar(top_items):
    try:
        import matplotlib.pyplot as plt
        labels = [w for w,c in top_items]
        counts = [c for w,c in top_items]
        fig, ax = plt.subplots()
        ax.bar(labels, counts)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        st.pyplot(fig)
    except Exception as e:
        st.info(f"Не вдалося побудувати barplot: {e}")


def run_lda(docs: List[str], n_topics:int=5, n_top_words:int=10):
    if not _HAS_SKLEARN:
        return None, "ERROR: sklearn не встановлено (pip install scikit-learn)"
    try:
        vectorizer = CountVectorizer(stop_words=list(STOPWORDS), max_df=0.95, min_df=2)
        X = vectorizer.fit_transform(docs)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X)
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            top_weights = topic[top_features_ind]
            topics.append((topic_idx, list(zip(top_features, top_weights))))
        return topics, None
    except Exception as e:
        return None, f"LDA failed: {e}"

# Streamlit app

def main():
    st.set_page_config(page_title="Текстова аналітика документів (без PyPDF2)", layout="wide")
    st.title("📝 Текстова аналітика документів — Streamlit (pdfminer)")

    st.markdown("Завантажте PDF або TXT. Додаток витягне текст, порахує частоту слів, побудує wordcloud і barplot та проведе тематичне моделювання (LDA).\n\nНе використовуємо PyPDF2 та pdfplumber.")

    st.sidebar.header("Налаштування")
    top_n = st.sidebar.number_input("Скільки топ-слів показувати?", min_value=5, max_value=100, value=20)
    n_topics = st.sidebar.number_input("Кількість тем (LDA)", min_value=2, max_value=20, value=5)
    n_top_words = st.sidebar.number_input("Скільки слів на тему (LDA)", min_value=3, max_value=20, value=8)

    uploaded = st.file_uploader("Завантажте PDF або TXT", type=["pdf","txt"] )
    if not uploaded:
        st.info("Завантажте файл у форматі PDF або TXT вгорі.")
        return

    file_bytes = uploaded.read()
    if uploaded.type == "application/pdf" or uploaded.name.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_bytes)
    else:
        try:
            text = file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text = file_bytes.decode('cp1251')
            except Exception:
                text = file_bytes.decode('utf-8', errors='ignore')

    if text.startswith("ERROR"):
        st.error(text)
        return

    text = clean_text(text)
    if not text:
        st.warning("Не вдалося витягти текст або текст порожній.")
        return

    st.subheader("Огляд документа")
    words = tokenize_words(text)
    chars = len(text)
    unique_words = len(set(words))
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Символів", f"{chars}")
        st.metric("Слів", f"{len(words)}")
    with col2:
        st.metric("Унікальних слів", f"{unique_words}")
        st.metric("Рядків (прибл.)", f"{text.count('\\n')+1}")
    with col3:
        if _HAS_PDFMINER:
            st.success("pdfminer.six: доступний")
        else:
            st.warning("pdfminer.six: відсутній — PDF-обробка не працюватиме")
        if _HAS_WORDCLOUD:
            st.success("wordcloud: доступний")
        else:
            st.info("wordcloud: відсутній — хмара слів не збудується")
        if _HAS_SKLEARN:
            st.success("scikit-learn: доступний")
        else:
            st.info("scikit-learn: відсутній — LDA не працюватиме")

    st.markdown("---")
    st.subheader("Частотні слова")
    top_words = get_top_n(words, n=top_n)
    st.table([{"word": w, "count": c} for w,c in top_words])

    st.subheader("Візуалізації")
    vis_col1, vis_col2 = st.columns(2)
    with vis_col1:
        st.write("**Word Cloud**")
        if _HAS_WORDCLOUD:
            # Генеруємо текст на основі частот для кращої хмари
            wc_text = ' '.join([w for w,c in top_words for _ in range(c)])
            plot_wordcloud(wc_text)
        else:
            st.info("Встановіть wordcloud: pip install wordcloud")
    with vis_col2:
        st.write("**Barplot — топ слів**")
        plot_bar(top_words[:20])

    st.markdown("---")
    st.subheader("Тематичне моделювання (LDA)")
    # Для LDA розіб'ємо документ на 'документи' — простий підхід: розбити по абзацам/частинам
    docs = [p.strip() for p in text.split('\n\n') if len(p.strip())>50]
    if not docs:
        # fallback: розбити на шматки по N символів
        chunk_size = 2000
        docs = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    topics, error = run_lda(docs, n_topics=n_topics, n_top_words=n_top_words)
    if error:
        st.error(error)
    else:
        for tid, terms in topics:
            st.markdown(f"**Тема {tid+1}**: " + ", ".join([t for t,w in terms]))

    st.markdown("---")
    st.caption("Примітка: для кращих результатів LDA краще подавати набір документів (корпус). Для одного великого документу використовуйте розбиття на логічні частини або абзаци.")

if __name__ == '__main__':
    main()
