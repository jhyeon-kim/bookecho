# db_connector.py

from flask import g
import os
import pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

def connect_to_pinecone():
    load_dotenv()
    pinecone_api_key = os.getenv('PINECONE_API_KEY')

    pinecone.init(api_key=pinecone_api_key, environment='asia-southeast1-gcp-free')
    g.model = SentenceTransformer('all-MiniLM-L6-v2')
    g.index = pinecone.Index("book-echo")
