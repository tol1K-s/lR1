import streamlit as st
import PyPDF2
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from gensim import corpora, models

nltk.download('punkt')
nltk.download('stopwords')

st.title("📊 Текстова аналітика документів")

uploaded_file = st.file_uploader("Завантаж PDF або TXT", type=["pdf", "txt"])

if uploaded_file:
    # --- Витяг тексту ---
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        text = " ".join(page.extract_text() for page in reader.pages)
    else:
        text = uploaded_file.read().decode("utf-8")

    st.subheader("📄 Вміст тексту:")
    st.text_area("", text[:1000] + "...", height=200)

    # --- Обробка тексту ---
    tokens = nltk.word_tokenize(text.lower())
    words = [w for w in tokens if w.isalpha() and w not in stopwords.words("english")]
    freq = nltk.FreqDist(words)

    # --- Частотна діаграма ---
    st.subheader("🔢 Частота слів")
    top_words = freq.most_common(20)
    sns.barplot(x=[w for w, _ in top_words], y=[c for _, c in top_words])
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())

    # --- WordCloud ---
    st.subheader("☁️ Хмара слів")
    wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(words))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt.gcf())

    # --- Тематичне моделювання (LDA) ---
    st.subheader("🧩 Тематичне моделювання (LDA)")
    dictionary = corpora.Dictionary([words])
    corpus = [dictionary.doc2bow(words)]
    lda = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

    for i, topic in lda.show_topics(num_topics=3, formatted=False):
        st.write(f"**Тема {i+1}:**", ", ".join([w for w, _ in topic]))
