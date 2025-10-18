import streamlit as st
import io
import re
from collections import Counter
from typing import List, Tuple

try:
from sklearn.feature_extraction.text import TfidfVectorizer
_HAS_SKLEARN = True
except Exception:
_HAS_SKLEARN = False


try:
from wordcloud import WordCloud
import matplotlib.pyplot as plt
_HAS_WORDCLOUD = True
except Exception:
_HAS_WORDCLOUD = False


try:
from textblob import TextBlob
_HAS_TEXTBLOB = True
except Exception:
_HAS_TEXTBLOB = False
