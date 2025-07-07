from langchain.chat_models import SmartAssistant
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm = SmartAssistant(model_name="gpt-3.5-turbo", temperature=0)

def generate_summary(document_text):
    prompt = PromptTemplate(
        input_variables=["content"],
        template="Summarize the following document in less than 150 words:\n\n{content}"
    )
    return llm.predict(prompt.format(content=document_text[:3000]))
