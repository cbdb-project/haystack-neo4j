from haystack import Document
# from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from neo4j_haystack import Neo4jDocumentStore
import time
from config import model_name, dimention
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.utils import Secret


#Time spend in minutes:  931.9594766457875

document_store = Neo4jDocumentStore(
    url="bolt://localhost:7687",
    username="haystack",
    password="haystack",
    database="haystack",
    embedding_dim=dimention,
    index="document-embeddings",
)

input_text = []

print("1. Reading data from file")
time_checkpoint = time.time()
with open('data.txt', 'r', encoding="utf-8") as file:
    input_text = file.readlines()
print("Time spend in minutes: ", (time.time() - time_checkpoint)/60)

# Sampling the data
input_text = input_text[:10]

print("2. Data read from file")
time_checkpoint = time.time()
documents = [Document(content=text) for text in input_text]
print("Time spend in minutes: ", (time.time() - time_checkpoint)/60)

# documents = [
#     Document(content="My name is Morgan and I live in Paris.", meta={"release_date": "2018-12-09"})]

with open("api_token.txt", "r") as file:
    api_token = file.read().strip()
print("3. Embedding documents")
time_checkpoint = time.time()
document_embedder = OpenAIDocumentEmbedder(api_key=Secret.from_token(api_token), model=model_name)
print("Time spend in minutes: ", (time.time() - time_checkpoint)/60)

# print("4. Warming up")
# time_checkpoint = time.time()
# document_embedder.warm_up()
# print("Time spend in minutes: ", (time.time() - time_checkpoint)/60)

print("5. Running")
time_checkpoint = time.time()
documents_with_embeddings = document_embedder.run(documents)
print("Time spend in minutes: ", (time.time() - time_checkpoint)/60)

print("6. Writing documents to Neo4j")
time_checkpoint = time.time()
document_store.write_documents(documents_with_embeddings.get("documents"))
print("Time spend in minutes: ", (time.time() - time_checkpoint)/60)