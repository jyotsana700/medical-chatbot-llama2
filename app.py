from src.helper import main
from flask import Flask,jsonify,render_template,request
from src import *
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


app=Flask(__name__)

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="demo_collection",
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="demo_collection",
    embedding=embedding,
)
