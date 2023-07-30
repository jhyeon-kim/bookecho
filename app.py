# app.py

from flask import Flask, render_template, request, jsonify, g
from db_connector import connect_to_pinecone

app = Flask(__name__)


@app.before_request
def before_request():
    connect_to_pinecone()


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/upsert', methods=['POST', 'GET'])
def upsert_data():
    if request.method == 'POST':
        return upsert_post()
    return render_template('upsert.html')


@app.route('/query', methods=['GET', 'POST'])
def query_data():
    if request.method == 'POST':
        return query_post()
    return render_template('query.html')


@app.route('/list', methods=['GET'])
def list_data():
    return list_get()


# Upsert and Query POST methods
def upsert_post():
    data = request.get_json()  # Get the JSON data sent in the request
    if 'data' not in data:
        return jsonify({"message": "No data provided"}), 400

    sentences = data['data']  # Extract the sentences to be inserted
    upsert_data(sentences)
    return jsonify({"message": "Data upserted successfully"}), 200


def query_post():
    data = request.get_json()  # Get the JSON data sent in the request
    if 'query' not in data:
        return jsonify({"message": "No query provided"}), 400

    query = data['query']  # Extract the query
    print(f"🔍 query: {query}")
    query_em = g.model.encode(query).tolist()
    result = g.index.query(query_em, top_k=1, includeMetadata=True)
    result_data = result.get('matches')[0].get('metadata').get('content')
    print(f"😵‍💫 response: {result_data}")
    return {"response": result_data}, 200


# Upsert helper function
def upsert_data(sentences):
    upserted_data = []
    for i, sentence in enumerate(sentences):
        id = g.index.describe_index_stats()['total_vector_count']
        upserted_data.append(
            (
                str(id + i),
                g.model.encode(sentence).tolist(),
                {
                    'content': sentence
                }
            )
        )
    g.index.upsert(vectors=upserted_data)


def list_get():
    fetch = g.index.fetch(ids=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    print(f"fetch: {fetch}")
    return jsonify(fetch), 200


if __name__ == '__main__':
    app.run(debug=True)
