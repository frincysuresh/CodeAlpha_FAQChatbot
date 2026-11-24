import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="FAQ Chatbot", page_icon="💬")

st.title("💬 Python FAQ Chatbot")
st.write("Ask me questions about Python basics!")

faqs = {
    "What is Python?": "Python is a high-level, interpreted programming language.",
    "What are loops?": "Loops are used to repeat a block of code.",
    "What is a variable?": "A variable is a name used to store data.",
    "What is a function?": "A function is a reusable block of code.",
    "What is a list?": "A list is an ordered and changeable collection of items.",
    "What is a tuple?": "A tuple is an ordered and unchangeable collection of items.",
    "What is a dictionary?": "A dictionary stores data in key-value pairs.",
    "What is a string?": "A string is a sequence of characters.",
    "What is an integer?": "An integer is a whole number.",
    "What is a comment?": "A comment is text in code ignored by Python."
}

questions = list(faqs.keys())
answers = list(faqs.values())

user_question = st.text_input("Ask your question:")

if st.button("Get Answer"):
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        vectors = vectorizer.fit_transform(
            questions + [user_question]
        )

        similarity = cosine_similarity(
            vectors[-1],
            vectors[:-1]
        )

        best_match = similarity.argmax()
        score = similarity[0][best_match]

        if score >= 0.30:
            st.success(answers[best_match])
        else:
            st.info(
                "Sorry, I don't have a close FAQ match. "
                "Try asking about Python basics."
            )