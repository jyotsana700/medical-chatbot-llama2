from src.helper import *
from src.prompt import *
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import ctransformers
from pinecone import Pinecone
# from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from flask import Flask,jsonify,render_template,request

app=Flask(__name__)

extracted_data=load_pdf("data/")
text_chunks=text_split(extracted_data)
embedding=download_hugging_face_embedding(text_chunks)

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams

# Step 1: Connect to the Qdrant server (replace with your own host and port if necessary)
client = QdrantClient(
    url="https://6389e072-7adf-4098-8885-5e712126420f.europe-west3-0.gcp.cloud.qdrant.io:6333",  # Replace with your Qdrant instance URL
    api_key="yqowznzutXSi_9GIE2iDwC0w46WW5EIFvIVR1jVlyonF2zk36qxMAw"  # Replace with your actual API key
)

index='medical-cha1bot'

doc= client.search(index,embedding)
PROMPT=PromptTemplate(template=prompt_template,input_variables=["context","question"])

chain_type_kwargs={"prompt":PROMPT}

llm=ctransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_token':512,
                          "temperature":0.8})

qa=RetrievalQA.from_chain_type(llm=llm,chain_type='stuff',retriever=doc.as_retriever(search_kwargs={'k':2}),return_source_document=True,chain_type_kwargs=chain_type_kwargs)


@app.route("/")
def index():
    return render_template('chat.html')


if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080,debug=True)



