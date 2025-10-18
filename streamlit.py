import streamlit as st
import pandas as pd
import nltk
from nltk.corpus 
import stopwords
from wordcloud 
import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from gensim import corpora, models

# --- Налаштування ---
nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="Текстова аналітика документів", page_icon="📊", layout="wide")
st.title("📊 Текстова аналітика документів")

# --- Завантаження файлу ---
uploaded_file = st.file_uploader("📂 Завантаж PDF або TXT файл", type=["pdf", "txt"])

if uploaded_file:
    # --- Витяг тексту ---
    text = ""
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    else:
        text = uploaded_file.read().decode("utf-8")

    # --- Відображення уривку ---
    st.subheader("📄 Вміст тексту:")
    st.text_area("Попередній перегляд:", text[:1000] + "...", height=200)

    # --- Токенізація та очищення ---
    tokens = nltk.word_tokenize(text.lower())
    words = [w for w in tokens if w.isalpha() and w not in stopwords.words("english")]

    if len(words) < 5:
        st.warning("⚠️ У тексті недостатньо слів для аналізу.")
    else:
        # --- Частота слів через Pandas ---
        freq_dist = nltk.FreqDist(words)
        freq_df = pd.DataFrame(freq_dist.most_common(20), columns=["Слово", "Частота"])

        st.subheader("🔢 Частота слів")
        st.dataframe(freq_df)

        # --- Візуалізація частоти ---
        plt.figure(figsize=(10, 5))
        sns.barplot(data=freq_df, x="Слово", y="Частота", palette="viridis")
        plt.xticks(rotation=45)
        plt.title("Найчастотніші слова")
        st.pyplot(plt.gcf())

        # --- WordCloud ---
        st.subheader("☁️ Хмара слів")
        wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(words))
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt.gcf())

        # --- Тематичне моделювання (LDA) ---
        st.subheader("🧩 Тематичне моделювання (LDA)")
        dictionary = corpora.Dictionary([words])
        corpus = [dictionary.doc2bow(words)]
        lda = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

        topics = []
        for i, topic in lda.show_topics(num_topics=3, formatted=False):
            topic_words = ", ".join([w for w, _ in topic])
            topics.append({"Тема": f"Тема {i+1}", "Ключові слова": topic_words})

        topics_df = pd.DataFrame(topics)
        st.dataframe(topics_df)

        # --- Кнопка експорту ---
        csv = freq_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Завантажити частоту слів (CSV)",
            data=csv,
            file_name="word_frequency.csv",
            mime="text/csv",
        )
else:
    st.info("👆 Завантажте PDF або TXT файл для початку аналізу.")

