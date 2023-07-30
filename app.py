from flask import Flask, render_template
import pinecone
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load the .env file
load_dotenv()

app = Flask(__name__)

pinecone_api_key = os.getenv('PINECONE_API_KEY')

pinecone.init(api_key=pinecone_api_key, environment='asia-southeast1-gcp-free')
model = SentenceTransformer('all-MiniLM-L6-v2')

index = pinecone.Index("book-echo")

text_data = [
    "1983년, 과학자이자 작가인 론 화이트헤드는 <미국 물리학 저널>을 통해 도미노 하나가 줄지어 선 다른 도미노를 쓰러뜨릴 뿐만 아니라 훨씬 더 큰 것도 쓰러뜨릴 수 있다고 말했다. 구체적으로 한 개의 도미노는 자신보다 1.5배가 큰 것도 넘어뜨릴 수 있는 힘을 가진다로 그는 설명했다.",
    "정말 가치있는 무언가를 향해 잘 align되어 있자. 마치 잘 줄 세워놓은 도미노처럼 그 방향과 초점이 군더더기 없이 명료하다면, 나의 한 가지 실행은 작아보일지라도, 결국 큰 가치를 실현하도록 나보다 더 큰 힘들이 조응해줄 수 있다는 것을 기대하자."
]
#
# upserted_data = []
# i=0
# for item in text_data:
#   id  = index.describe_index_stats()['total_vector_count']
#   upserted_data.append(
#       (
#         str(id+i),
#         model.encode(item).tolist(),
#         {
#             'content': item
#         }
#       )
#   )
#   i+=1
# index.upsert(vectors=upserted_data)


query = "요즘 집중에 대해 고민이야. 관련있는 글 추천해줄래?"
query_em = model.encode(query).tolist()
result = index.query(query_em, top_k=1, includeMetadata=True)
print(result)


@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
