import os
from pathlib import Path

import streamlit as st

# Бібліотеки для обробки тексту та візуалізацій
import re
import string
from collections import Counter

import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# LDA (Gensim)
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess

import pyLDAvis
import pyLDAvis.gensim_models

# Ініціалізація NLTK
nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="Text Analytics App", layout="wide")

st.title("Текстова аналітика документів")

# 1) Завантаження файлу
uploaded_file = st.file_uploader("Завантаж PDF або TXT", type=["pdf", "txt"])

text = ""
if uploaded_file is not None:
    # Зчитування тексту залежно від типу
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        pages = [p.extract_text() or "" for p in pdf_reader.pages]
        text = "\n".join(pages)
    elif suffix == ".txt":
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    st.success("Текст успішно витягнуто.")

# Якщо текст є, продовжуємо
if text:
    # 2) Очистка та токенізація, частотність
    lang = st.selectbox("Мова обробки (для стоп-слів)", ["english", "russian", "ukrainian", "none"])
    if lang == "english":
        stops = set(stopwords.words("english"))
    elif lang == "russian":
        try:
            stops = set(stopwords.words("russian"))
        except:
            stops = set()
            st.warning("Список зупинок для російської може бути неповним.")
    elif lang == "ukrainian":
        # Якщо потрібні українські стоп-слова, їх можна підключити окремо
        stops = set(stopwords.words("english"))  # простіша заміна
        st.info("Українські стоп-слова не встановлені за замовчуванням. Використано англо-словник як базу.")
    else:
        stops = set()

    # Прості попередня обробка
    def tokenize_and_filter(text, language_stops):
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalpha()]
        if language_stops:
            tokens = [t for t in tokens if t not in language_stops]
        return tokens

    tokens = tokenize_and_filter(text, stops)
    freq = Counter(tokens)

    # 3) Візуалізації
    st.subheader("Word Cloud")
    wc = WordCloud(width=800, height=400, background_color="white", max_words=200)
    wc.generate_from_frequencies(freq)
    fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")

    st.pyplot(fig_wc)

    st.subheader("Top слів (barplot)")
    top_n = st.slider("Кількість топ слів", min_value=5, max_value=50, value=20, step=1)
    common = freq.most_common(top_n)
    words, counts = zip(*common) if common else ([], [])
    fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
    ax_bar.bar(words, counts)
    ax_bar.set_xticklabels(words, rotation=45, ha="right")
    ax_bar.set_xlabel("Слова")
    ax_bar.set_ylabel("Частота")
    st.pyplot(fig_bar)

    # 4) LDA-моделювання
    st.sidebar.header("LDA параметри")
    num_topics = st.sidebar.number_input("Кількість тем", min_value=2, max_value=20, value=5, step=1)
    passes = st.sidebar.number_input("Passes (циклів навчання)", min_value=1, max_value=20, value=10, step=1)

    # Розбиття документів на окремі документи для LDA
    # Якщо хочеш, можеш розділяти за абзацами. Тут зробимо один документ.
    docs = [text]

    # Токенізація документів
    tokenized_docs = [simple_preprocess(doc, deacc=True) for doc in docs]

    # Створення словника та корпусу
    dictionary = corpora.Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    if len(dictionary) > 0 and len(corpus) > 0:
        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=passes,
            random_state=42
        )

        topics = lda.print_topics(num_words=8)
        st.subheader("Теми (Top words per topic)")
        for t in topics:
            st.write(t)

        # Візуалізація pyLDAvis
        vis = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary)
        pyLDAvis.save_html(vis, "lda_visualization.html")
        with open("lda_visualization.html", "r", encoding="utf-8") as f:
            html_data = f.read()
        st.components.v1.html(html_data, height=600, width="100%")
else:
    st.info("Зачекайте завантаження файлу.")
