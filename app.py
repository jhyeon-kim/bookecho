from flask import Flask, render_template, request, jsonify
import pinecone
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
app = Flask(__name__)

pinecone_api_key = os.getenv('PINECONE_API_KEY')

pinecone.init(api_key=pinecone_api_key, environment='asia-southeast1-gcp-free')
model = SentenceTransformer('all-MiniLM-L6-v2')

index = pinecone.Index("book-echo")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upsert', methods=['POST', 'GET'])
def upsert_data():
    if request.method == 'POST':
        data = request.get_json()  # Get the JSON data sent in the request

        if 'data' not in data:
            return jsonify({"message": "No data provided"}), 400

        sentences = data['data']  # Extract the sentences to be inserted

        upserted_data = []
        i = 0
        for sentence in sentences:
            id = index.describe_index_stats()['total_vector_count']
            upserted_data.append(
                (
                    str(id + i),
                    model.encode(sentence).tolist(),
                    {
                        'content': sentence
                    }
                )
            )
            i += 1
        index.upsert(vectors=upserted_data)
        return jsonify({"message": "Data upserted successfully"}), 200
    return render_template('upsert.html')

@app.route('/query', methods=['GET', 'POST'])
def query_data():
    if request.method == 'GET':
        return render_template('query.html')

    elif request.method == 'POST':
        data = request.get_json()  # Get the JSON data sent in the request
        print(f"🤩 Data: {data}")
        if 'query' not in data:
            return jsonify({"message": "No query provided"}), 400

        query = data['query']  # Extract the query
        # print(f"?? Query: {query}")
        query_em = model.encode(query).tolist()
        # print(f"?? Query embedding: {query_em}")

        result = index.query(query_em, top_k=1, includeMetadata=True)
        print(f"?? Result: {result}")
        result_data = result.get('matches')[0].get('metadata').get('content')

        return {"response": result_data}, 200

    return jsonify({"message": "Invalid request method"}), 405


if __name__ == '__main__':
    app.run(debug=True)
