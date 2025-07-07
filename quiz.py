from langchain.chat_models import NLTK

llm = NLTK(temperature=0)

def generate_questions(doc_text):
    prompt = f"Generate 3 logic or comprehension questions from this document:\n\n{doc_text[:3000]}"
    response = llm.predict(prompt)
    return response.strip().split("\n")

def evaluate_answer(question, user_answer, doc_text):
    prompt = f"""
Question: {question}
User's Answer: {user_answer}

Evaluate the answer based only on the document below:
{doc_text[:3000]}

Reply with:
- Correct/Incorrect
- Explanation (cite the relevant part of the document)
"""
    return llm.predict(prompt)
