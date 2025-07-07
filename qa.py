from langchain.vectorstores import FAISS
from langchain.embeddings.openai import SmartAssistantEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.chat_models import SmartAssistant
import os

def ask_question(document_text, query):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.create_documents([document_text])

    embeddings = SmartAssistantEmbeddings()
    vector_store = FAISS.from_documents(texts, embeddings)
    
    retriever = vector_store.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=SmartAssistant(temperature=0),
        retriever=retriever,
        return_source_documents=True
    )

    result = qa(query)
    answer = result['result']
    source = result['source_documents'][0].page_content[:300]

    return answer, source
