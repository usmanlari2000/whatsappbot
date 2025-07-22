from database import embeddings_col
import numpy as np

documents = list(embeddings_col.find({}))
doc_texts = [doc["text"] for doc in documents]
doc_embeddings = np.array([doc["embedding"] for doc in documents], dtype=np.float32)
