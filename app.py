import streamlit as st
from logic.loader import load_pdf
from logic.summarizer import generate_summary
from logic.qa import ask_question
from logic.quiz import generate_questions, evaluate_answer

st.title("📄 Smart Research Assistant")

uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    doc_text = load_pdf("temp.pdf")
    
    st.subheader("🔍 Auto Summary")
    summary = generate_summary(doc_text)
    st.write(summary)

    mode = st.radio("Select Mode", ["Ask Anything", "Challenge Me"])

    if mode == "Ask Anything":
        question = st.text_input("Enter your question:")
        if question:
            with st.spinner("Thinking..."):
                answer, justification = ask_question(doc_text, question)
            st.success(answer)
            st.caption(f"📌 Justified from: \"{justification}\"")

    elif mode == "Challenge Me":
        if st.button("Generate Questions"):
            questions = generate_questions(doc_text)
            for idx, q in enumerate(questions):
                st.markdown(f"**Q{idx+1}: {q}**")
                user_answer = st.text_input(f"Your Answer to Q{idx+1}")
                if user_answer:
                    feedback = evaluate_answer(q, user_answer, doc_text)
                    st.info(feedback)
