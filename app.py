"""
app.py
Streamlit UI for the NLP Text Preprocessing Toolbox.
Run with: streamlit run app.py
"""

import streamlit as st
from nlp_utils import (
    tokenize_text,
    remove_stopwords,
    stem_and_lemmatize,
    normalize_text,
    apply_regex,
    min_edit_distance
)

st.set_page_config(page_title="NLP Text Preprocessing Toolbox", layout="centered")

st.title("NLP Text Preprocessing Toolbox")
st.caption("Unit 1: Introduction to NLP — demonstrated with Streamlit")

# Sidebar navigation
module = st.sidebar.radio(
    "Select a module:",
    (
        "Tokenization",
        "Stop Word Removal",
        "Stemming & Lemmatization",
        "Text Normalization",
        "Regular Expressions",
        "Minimum Edit Distance"
    )
)

st.sidebar.markdown("---")
st.sidebar.caption("NLP Course Project — Unit 1")


# ---------- 1. TOKENIZATION ----------
if module == "Tokenization":
    st.header("Tokenization")
    text = st.text_area("Enter text:", "Natural Language Processing is amazing. It has many applications.")

    if st.button("Tokenize"):
        word_tokens, sentence_tokens = tokenize_text(text)
        st.subheader("Word Tokens")
        st.write(word_tokens)
        st.subheader("Sentence Tokens")
        st.write(sentence_tokens)


# ---------- 2. STOP WORD REMOVAL ----------
elif module == "Stop Word Removal":
    st.header("Stop Word Removal")
    text = st.text_area("Enter text:", "This is a simple example to show how stop words are removed from text.")

    if st.button("Remove Stop Words"):
        original, filtered = remove_stopwords(text)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Before")
            st.write(original)
        with col2:
            st.subheader("After")
            st.write(filtered)


# ---------- 3. STEMMING & LEMMATIZATION ----------
elif module == "Stemming & Lemmatization":
    st.header("Stemming & Lemmatization")
    text = st.text_area("Enter text:", "The children were running and playing happily in the fields.")

    if st.button("Process"):
        results = stem_and_lemmatize(text)
        st.subheader("Comparison Table")
        st.table(
            {
                "Original Word": [r[0] for r in results],
                "Stemmed": [r[1] for r in results],
                "Lemmatized": [r[2] for r in results],
            }
        )


# ---------- 4. TEXT NORMALIZATION ----------
elif module == "Text Normalization":
    st.header("Text Normalization")
    text = st.text_area("Enter text:", "Hello World!! This IS an Example... of Text-Normalization.")

    if st.button("Normalize"):
        original, lowered, no_punct = normalize_text(text)
        st.subheader("Original")
        st.write(original)
        st.subheader("Lowercased")
        st.write(lowered)
        st.subheader("Lowercased + Punctuation Removed")
        st.write(no_punct)


# ---------- 5. REGULAR EXPRESSIONS ----------
elif module == "Regular Expressions":
    st.header("Regular Expressions")
    text = st.text_area(
        "Enter text:",
        "Contact us at support@example.com or admin@test.org. Call 9876543210. Date: 25-09-2026."
    )
    pattern = st.text_input("Enter a custom regex pattern (leave empty for auto-extraction of emails/numbers/dates):")

    if st.button("Apply Regex"):
        results = apply_regex(text, pattern)
        for key, value in results.items():
            st.subheader(key)
            st.write(value)


# ---------- 6. MINIMUM EDIT DISTANCE ----------
elif module == "Minimum Edit Distance":
    st.header("Minimum Edit Distance")
    col1, col2 = st.columns(2)
    with col1:
        word1 = st.text_input("Word 1:", "kitten")
    with col2:
        word2 = st.text_input("Word 2:", "sitting")

    if st.button("Compute Distance"):
        distance, dp_table = min_edit_distance(word1, word2)
        st.session_state["med_distance"] = distance
        st.session_state["med_dp_table"] = dp_table

    if "med_distance" in st.session_state:
        st.subheader(f"Minimum Edit Distance: {st.session_state['med_distance']}")

        show_table = st.checkbox("Show DP table", key="dp_table_toggle")
        if show_table:
            st.subheader("DP Table")
            st.write(st.session_state["med_dp_table"])