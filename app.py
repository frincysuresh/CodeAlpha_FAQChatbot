
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="FAQ Chatbot", page_icon="💬")

st.title("💬 Python FAQ Chatbot")
st.write("Ask questions about Python programming!")

faqs = [
    {
        "questions": ["what is python", "define python", "explain python"],
        "answer": "Python is a high-level, interpreted programming language known for its simple syntax."
    },
    {
        "questions": ["what are loops in python", "what is a loop", "explain loops"],
        "answer": "Loops are used to repeat a block of code. Python has for loops and while loops."
    },
    {
        "questions": ["what is a variable", "define variable", "variables in python"],
        "answer": "A variable is a name used to store data in a program."
    },
    {
        "questions": ["what is a function", "define function", "functions in python"],
        "answer": "A function is a reusable block of code that performs a specific task."
    },
    {
        "questions": ["what is a list", "explain list"],
        "answer": "A list is an ordered and changeable collection of items in Python."
    },
    {
        "questions": ["what is a tuple", "explain tuple"],
        "answer": "A tuple is an ordered collection of items that cannot be changed."
    },
    {
        "questions": ["what is a dictionary", "explain dictionary"],
        "answer": "A dictionary stores data in key-value pairs."
    },
    {
        "questions": ["what is a string", "explain string"],
        "answer": "A string is a sequence of characters enclosed in quotes."
    },
    {
        "questions": ["what is an operator", "explain operators"],
        "answer": "Operators are symbols used to perform operations such as addition and comparison."
    },
    {
        "questions": ["what is indentation", "explain indentation"],
        "answer": "Indentation means spaces at the beginning of a line. Python uses it to define code blocks."
    }
]

questions = []
answers = []

for faq in faqs:
    for question in faq["questions"]:
        questions.append(question)
        answers.append(faq["answer"])

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
faq_vectors = vectorizer.fit_transform(questions)

user_question = st.text_input("Ask your question")

if st.button("Submit"):
    if user_question.strip():
        user_vector = vectorizer.transform([user_question])
        scores = cosine_similarity(user_vector, faq_vectors)[0]
        best_match = scores.argmax()

        if scores[best_match] >= 0.15:
            st.subheader("Answer")
            st.success(answers[best_match])
        else:
            st.warning("Sorry, I don't have a close FAQ match. Try asking about Python basics.")
    else:
        st.warning("Please enter a question.")

st.caption("Built using Python, Streamlit and TF-IDF.")
