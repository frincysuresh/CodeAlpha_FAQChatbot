        question_vector = vectorizer.transform([user_question])
        scores = cosine_similarity(question_vector, faq_vectors)[0]

        best_match = scores.argmax()

        if scores[best_match] >= 0.25:
            st.success(faq_answers[best_match])
        else:
            st.warning("Sorry, I don't have a close FAQ match. Try asking about Python basics.")
    else:
        st.warning("Please enter a question.")