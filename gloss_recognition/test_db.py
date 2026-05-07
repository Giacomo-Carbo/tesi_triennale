import os
import numpy as np
import tensorflow as tf
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import random
import json
# --- 1. CONFIGURAZIONE ---
client = QdrantClient(url="http://localhost:6333")
MODEL_PATH = "final_lstm_encoder.keras"
DATA_PATH = "MP_DATA_EMBEDDINGS"


encoder = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)

def extract_embedding(file_path):
    data = np.load(file_path)
    data = np.expand_dims(data, axis=0)
    embedding = encoder.predict(data, verbose=0)
    return embedding.flatten().tolist()


print("Eseguo query di test\n\n")

# Esempio di query: prendo un file a caso e cerco i 5 più simili
random_gloss = random.choice(os.listdir(DATA_PATH))
random_gloss_dir = os.path.join(DATA_PATH, random_gloss)
random_file = random.choice([f for f in os.listdir(random_gloss_dir) if f.endswith(".npy")])
query_path = os.path.join(random_gloss_dir, random_file)
query_vector = extract_embedding(query_path)    
search_result = client.query_points(
    collection_name="gloss",
    query=query_vector,
    limit=5
)
print(random_gloss)
print(f"gloss predetto {search_result.points[0].payload['gloss']} con distanza {search_result.points[0].score}")