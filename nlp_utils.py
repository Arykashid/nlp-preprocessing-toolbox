"""
nlp_utils.py
Core NLP logic functions for the NLP Text Preprocessing Toolbox.
Covers: Tokenization, Stop Word Removal, Stemming & Lemmatization,
Text Normalization, Regular Expressions, Minimum Edit Distance.
"""

import re
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data (only runs once, then cached)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


# ---------- 1. TOKENIZATION ----------
def tokenize_text(text):
    """Returns word tokens and sentence tokens separately."""
    word_tokens = word_tokenize(text)
    sentence_tokens = sent_tokenize(text)
    return word_tokens, sentence_tokens


# ---------- 2. STOP WORD REMOVAL ----------
def remove_stopwords(text):
    """Removes English stopwords from the text. Returns before/after lists."""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word.lower() not in stop_words]
    return tokens, filtered


# ---------- 3. STEMMING & LEMMATIZATION ----------
def stem_and_lemmatize(text):
    """Returns a list of (word, stemmed, lemmatized) tuples for comparison."""
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)

    results = []
    for word in tokens:
        if word.isalpha():  # skip punctuation
            stemmed = stemmer.stem(word)
            lemmatized = lemmatizer.lemmatize(word.lower())
            results.append((word, stemmed, lemmatized))
    return results


# ---------- 4. TEXT NORMALIZATION ----------
def normalize_text(text):
    """Lowercases text and removes punctuation. Returns before/after."""
    original = text
    lowered = text.lower()
    no_punct = lowered.translate(str.maketrans('', '', string.punctuation))
    return original, lowered, no_punct


# ---------- 5. REGULAR EXPRESSIONS ----------
def apply_regex(text, pattern):
    """Applies a user-given regex pattern to text and returns matches.
    If pattern is empty, runs default extraction (emails, numbers, dates)."""
    if pattern.strip() == "":
        emails = re.findall(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}', text)
        numbers = re.findall(r'\d+', text)
        dates = re.findall(r'\d{2}[-/]\d{2}[-/]\d{4}', text)
        return {"Emails": emails, "Numbers": numbers, "Dates": dates}
    else:
        try:
            matches = re.findall(pattern, text)
            return {"Matches": matches}
        except re.error as e:
            return {"Error": str(e)}


# ---------- 6. MINIMUM EDIT DISTANCE ----------
def min_edit_distance(word1, word2):
    """Computes Levenshtein (minimum edit) distance between two words
    using dynamic programming. Returns distance and the DP table."""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],     # deletion
                    dp[i][j - 1],     # insertion
                    dp[i - 1][j - 1]  # substitution
                )

    return dp[m][n], dp