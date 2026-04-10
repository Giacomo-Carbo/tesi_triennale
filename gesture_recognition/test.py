import os
import numpy as np
import tensorflow as tf
import chromadb
from chromadb.config import Settings
import random
# --- 1. CONFIGURAZIONE ---
MODEL_PATH = "final_lstm_encoder.h5"
DATA_PATH = "MP_DATA_EMBEDDINGS" # La cartella con le sottocartelle delle glosse
CHROMA_DB_DIR = "asl_vector_db"

# Carica l'encoder addestrato
encoder = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Inizializza ChromaDB (Persistente su disco)
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_or_create_collection(name="sign_language_embeddings")


# --- 2. FUNZIONE PER GENERARE EMBEDDING ---
def extract_embedding(file_path):
    data = np.load(file_path) # Carica (60, 1662)
    data = np.expand_dims(data, axis=0) # Diventa (1, 60, 1662)
    embedding = encoder.predict(data, verbose=0)
    return embedding.flatten().tolist()


print("\n🔍 Esempio di ricerca semantica...")

# prendo un video da quelli del dataset 
test_gloss = random.choice(os.listdir(DATA_PATH))
test_file = os.listdir(os.path.join(DATA_PATH, test_gloss))[0]
test_path = os.path.join(DATA_PATH, test_gloss, test_file)

query_vector = extract_embedding(test_path)

# Cerco i 3 segni più vicini nello spazio vettoriale
results = collection.query(
    query_embeddings=[query_vector],
    n_results=3
)

print(f"\nRisultati per il segno di test '{test_gloss}':")
for i in range(len(results['ids'][0])):
    print(f"{i+1}. Gloss: {results['metadatas'][0][i]['gloss']} (ID: {results['ids'][0][i]}) - Distanza: {results['distances'][0][i]}")