from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from neo4j_haystack import Neo4jEmbeddingRetriever, Neo4jDocumentStore
from typing import List
import time

document_store = Neo4jDocumentStore(
    url="bolt://localhost:7687",
    username="neo4j",
    password="passw0rd",
    database="neo4j",
    embedding_dim=384,
    embedding_field="embedding",
    index="document-embeddings", # The name of the Vector Index in Neo4j
    node_label="Document", # Providing a label to Neo4j nodes which store Documents
)

print("Document count: ", document_store.count_documents()) 
print("1, Start pipeline")
time_checkpoint = time.time()
pipeline = Pipeline()
print("Time spend in minutes: ", (time.time() - time_checkpoint_1)/60)
print("2. Add components: text_embedder")
time_checkpoint = time.time()
pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
print("Time spend in minutes: ", (time.time() - time_checkpoint)/60)
print("3. Add components: retriever")
time_checkpoint = time.time()
pipeline.add_component("retriever", Neo4jEmbeddingRetriever(document_store=document_store))
print("Time spend in minutes: ", (time.time() - time_checkpoint)/60)
pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

print("4. Running pipeline")
time_checkpoint = time.time()
result = pipeline.run(
    data={
        "text_embedder": {"text": "What cities do people live in?"},
        "retriever": {
            "top_k": 5,
            "filters": {"field": "release_date", "operator": "==", "value": "2018-12-09"},
        },
    }
)
print("Time spend in minutes: ", (time.time() - time_checkpoint)/60)

print("5. Extracting documents")
time_checkpoint = time.time()
documents: List[Document] = result["retriever"]["documents"]
print("Time spend in minutes: ", (time.time() - time_checkpoint)/60)
print("Documents: ", documents)
print("Document count: ", document_store.count_documents())
print("6. Done")