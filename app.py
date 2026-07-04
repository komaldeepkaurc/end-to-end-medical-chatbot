from flask import Flask, render_template, request, jsonify
from src.helper import download_hugging_face_embeddings, load_pdf_file, text_split
from langchain_pinecone import PineconeVectorStore, embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(index_name)

docsearch = PineconeVectorStore(
    index=index,
    embedding=embeddings
)
    
retriever = docsearch.as_retriever(search_type = "similarity",search_kwargs={"k": 3})
    
llm = ChatGroq(
    model="llama-3.3-70b-versatile",   # or another Groq-supported model
    temperature=0.4,
    max_tokens=500,
    api_key=GROQ_API_KEY
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.get_json()["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": input})
    print("Response: ", response["answer"])
    return str(response["answer"])


if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8080, debug=True)