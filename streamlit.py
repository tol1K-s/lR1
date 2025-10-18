import re
import string
from pathlib import Path
from collections import Counter

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import PyPDF2
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
import pyLDAvis.gensim_models
import pyLDAvis

# 1) Завантаження файлу
def extract_text_from_pdf(pdf_path):
    text = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            t = page.extract_text() or ""
            text.append(t)
    return "\n".join(text)

def load_text(path):
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(path)
    elif path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError("Unsupported file type")

# 2) Попередня обробка та частотність
def preprocess_text(text, lang='russian'):
    stop = set(stopwords.words('russian'))  # змінити на 'english' або відповідну мову
    # простий токенізатор
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]  # прибираємо цифри та знаки
    tokens = [t for t in tokens if t not in stop]
    return tokens

def word_frequencies(tokens):
    return Counter(tokens)

# 3) WordCloud та Barplot
def plot_wordcloud(freq, max_words=200, output_path="wordcloud.png"):
    wc = WordCloud(width=800, height=400, background_color="white", max_words=max_words)
    wc.generate_from_frequencies(freq)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_top_words_bar(freq, top_n=20, output_path="top_words.png"):
    common = freq.most_common(top_n)
    words, counts = zip(*common)
    plt.figure(figsize=(10,6))
    plt.bar(words, counts)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# 4) LDA з gensim
def lda_model_from_tokens(tokens_list, num_topics=5, passes=10):
    # створюємо словник та корпус
    dictionary = corpora.Dictionary([tokens_list])
    corpus = [dictionary.doc2bow(tokens_list)]
    # але нам потрібні багато документів; для прикладу використаємо список документів
    return dictionary, corpus

# Розширений приклад з кількома документами
def run_example(doc_texts, num_topics=5):
    # токени для кожного документа
    tokenized = [simple_preprocess(doc, deacc=True) for doc in doc_texts]
    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(text) for text in tokenized]

    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10)
    # топ-слова для кожної теми
    topics = lda.print_topics(num_words=10)
    for t in topics:
        print(t)

    # візуалізація
    vis = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(vis, "lda_visualization.html")

    return lda, dictionary, corpus

# Приклад використання
if __name__ == "__main__":
    path = "document.pdf"  # заміни на свій файл
    text = load_text(path)

    tokens = preprocess_text(text)
    freq = word_frequencies(tokens)

    plot_wordcloud(freq, output_path="wordcloud.png")
    plot_top_words_bar(freq, top_n=20, output_path="top_words.png")

    # Приклад для LDA: треба кілька документів
    docs = [
        "This is a sample document about data science and machine learning.",
        "Natural language processing and text analytics are part of data science.",
        "Statistics and probability are foundational to machine learning.",
        "Text analytics involves extracting topics and themes from documents."
    ]
    lda_model_from_tokens([tokenize for tokenize in [simple_preprocess(d) for d in docs]], num_topics=3)

    # або використати run_example з реальною парою документів
    lda, dictionary, corpus = run_example(docs, num_topics=3)
