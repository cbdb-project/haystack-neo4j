from typing import List

from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from neo4j_haystack import Neo4jEmbeddingRetriever, Neo4jDocumentStore
import time


document_store = Neo4jDocumentStore(
    url="bolt://localhost:7687",
    username="neo4j",
    password="passw0rd",
    database="neo4j",
    embedding_dim=384,
    index="document-embeddings",
)

input_text = []

print("1. Reading data from file")
time_checkpoint_1 = time.time()
with open('data.txt', 'r', encoding="utf-8") as file:
    input_text = file.readlines()
print("Time spend in minutes: ", (time.time() - time_checkpoint_1)/60)

print("2. Data read from file")
time_checkpoint_2 = time.time()
documents = [Document(content=text) for text in input_text]
print("Time spend in minutes: ", (time_checkpoint_2 - time_checkpoint_1)/60)

# documents = [
#     Document(content="My name is Morgan and I live in Paris.", meta={"release_date": "2018-12-09"})]

#
print("3. Embedding documents")
document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2") 
time_checkpoint_3 = time.time()
print("Time spend in minutes: ", (time_checkpoint_3 - time_checkpoint_2)/60)

print("4. Warming up")
document_embedder.warm_up()
time_checkpoint_4 = time.time()
print("Time spend in minutes: ", (time_checkpoint_4 - time_checkpoint_3)/60)

print("5. Running")
documents_with_embeddings = document_embedder.run(documents)
time_checkpoint_5 = time.time()
print("Time spend in minutes: ", (time_checkpoint_5 - time_checkpoint_4)/60)

print("6. Writing documents to Neo4j")
document_store.write_documents(documents_with_embeddings.get("documents"))
time_checkpoint_6 = time.time()
print("Time spend in minutes: ", (time_checkpoint_6 - time_checkpoint_5)/60)