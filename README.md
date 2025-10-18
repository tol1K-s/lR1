import streamlit as st
import pdfplumber
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('ukrainian') + stopwords.words('russian'))

st.title("📊 Текстова аналітика документів")

uploaded_file = st.file_uploader("Завантажте PDF або TXT файл", type=["pdf", "txt"])

def extract_text(file):
    if file.name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + " "
        return text
    else:
        return file.read().decode("utf-8")

if uploaded_file:
    text = extract_text(uploaded_file)
    st.subheader("📝 Витягнутий текст")
    st.write(text[:1000] + "..." if len(text) > 1000 else text)

    # Очистка тексту
    words = re.findall(r'\b\w+\b', text.lower())
    words = [w for w in words if w not in stop_words and len(w) > 2]

    # Частота слів
    freq = Counter(words)
    most_common = freq.most_common(20)

    st.subheader("📈 Частота найпоширеніших слів")
    st.bar_chart(dict(most_common))

    # WordCloud
    st.subheader("☁️ WordCloud")
    wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(words))
    st.image(wc.to_array())

    # LDA Тематичне моделювання
    st.subheader("🧩 Тематичне моделювання (LDA)")
    dictionary = corpora.Dictionary([words])
    corpus = [dictionary.doc2bow(words)]
    lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

    topics = lda_model.print_topics(num_words=5)
    for i, topic in topics:
        st.write(f"**Тема {i+1}:** {topic}")
