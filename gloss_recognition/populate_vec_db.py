import os
import numpy as np
import tensorflow as tf
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid 

# --- 1. CONFIGURAZIONE ---
client = QdrantClient(url="http://localhost:6333")
MODEL_PATH = "final_lstm_encoder.h5"
DATA_PATH = "MP_DATA_EMBEDDINGS"
COLLECTION_NAME = "gloss"
BATCH_SIZE = 100  

# Creazione collezione (con controllo se esiste già)
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=256, distance=Distance.EUCLID),
    )

encoder = tf.keras.models.load_model(MODEL_PATH, compile=False)

def extract_embedding(file_path):
    data = np.load(file_path)
    data = np.expand_dims(data, axis=0)
    embedding = encoder.predict(data, verbose=0)
    return embedding.flatten().tolist()

# --- 3. POPOLAMENTO A BATCH ---
print(f"Inizio indicizzazione in Qdrant (Batch Size: {BATCH_SIZE})...\n\n")

current_batch = []
total_count = 0

for gloss in os.listdir(DATA_PATH):
    gloss_dir = os.path.join(DATA_PATH, gloss)
    if not os.path.isdir(gloss_dir): continue
    
    print(f"Processando gloss: {gloss}")
    for file_name in os.listdir(gloss_dir):
        if not file_name.endswith(".npy"): continue
        
        file_path = os.path.join(gloss_dir, file_name)
        
        try:
            vector = extract_embedding(file_path)
            
            # Creiamo il punto
            point = PointStruct(
                id=str(uuid.uuid4()), # UUID è più sicuro per evitare collisioni
                vector=vector, 
                payload={"gloss": gloss, "file": file_name}
            )
            current_batch.append(point)
            
            # Se il batch è pieno, lo inviamo
            if len(current_batch) >= BATCH_SIZE:
                client.upsert(collection_name=COLLECTION_NAME, points=current_batch)
                total_count += len(current_batch)
                print(f"✅ Inviati {total_count} vettori...")
                current_batch = [] # Svuota il batch
                
        except Exception as e:
            print(f"❌ Errore sul file {file_path}: {e}")

# Invia l'ultimo batch rimanente (se non è vuoto)
if current_batch:
    client.upsert(collection_name=COLLECTION_NAME, points=current_batch)
    total_count += len(current_batch)
    print(f"🏁 Indicizzazione completata. Totale vettori: {total_count}")