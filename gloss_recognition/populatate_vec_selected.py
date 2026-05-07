import os
import numpy as np
import tensorflow as tf
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid 
from tqdm import tqdm

# --- 1. CONFIGURAZIONE ---
client = QdrantClient(url="http://localhost:6333")
MODEL_PATH = "final_lstm_encoder.keras"
DATA_PATH = "MP_DATA_EMBEDDINGS"
COLLECTION_NAME = "gloss"
BATCH_SIZE = 500  

# --- 2. FILTRO TARGET ---
# Inserisci qui solo i gloss che vuoi processare
target_glosses = [ "home","look","name","see","thank you","what","where","who","you","love", "lion", "hello"]

# Creazione collezione (con controllo se esiste già)
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=256, distance=Distance.COSINE),
    )

# Caricamento del modello
encoder = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)

def extract_embedding(file_path):
    data = np.load(file_path)
    data = np.expand_dims(data, axis=0)
    embedding = encoder.predict(data, verbose=0)
    return embedding.flatten().tolist()

# --- 3. POPOLAMENTO A BATCH ---
print(f"Inizio indicizzazione SELETTIVA per: {target_glosses}\n")

current_batch = []
total_count = 0

# Iteriamo solo sui gloss specificati invece di os.listdir()
for gloss in tqdm(target_glosses, desc="Processing Selected Gloss"):
    gloss_dir = os.path.join(DATA_PATH, gloss)
    
    # Verifichiamo se la cartella del gloss esiste effettivamente nel path
    if not os.path.isdir(gloss_dir):
        print(f"⚠️ Salto '{gloss}': Cartella non trovata in {DATA_PATH}")
        continue
    
    # Processiamo i file all'interno della cartella specifica
    for file_name in os.listdir(gloss_dir):
        if not file_name.endswith(".npy"): 
            continue
        
        file_path = os.path.join(gloss_dir, file_name)
        
        try:
            vector = extract_embedding(file_path)
            
            # Creiamo il punto per Qdrant
            point = PointStruct(
                id=str(uuid.uuid4()), 
                vector=vector, 
                payload={"gloss": gloss, "file": file_name}
            )
            current_batch.append(point)
            
            # Se il batch è pieno, lo inviamo
            if len(current_batch) >= BATCH_SIZE:
                client.upsert(collection_name=COLLECTION_NAME, points=current_batch)
                total_count += len(current_batch)
                current_batch = [] # Reset del batch
                
        except Exception as e:
            print(f"Errore sul file {file_path}: {e}")

# Invia l'ultimo batch rimanente (se non è vuoto)
if current_batch:
    client.upsert(collection_name=COLLECTION_NAME, points=current_batch)
    total_count += len(current_batch)

print(f"\nOperazione completata.")
print(f"Totale vettori inseriti per {target_glosses}: {total_count}")