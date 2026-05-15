from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from pydantic import BaseModel
from typing import List
import numpy as np
import tensorflow as tf
from scipy import interpolate
import os
import time
from tqdm import tqdm
import uvicorn
import uuid

# Configurazione environment
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "gloss")
MODEL_PATH = os.getenv("MODEL_PATH", "final_lstm_encoder.keras")
DATA_PATH = os.getenv("DATA_PATH", "MP_DATA_EMBEDDINGS")
target_glosses = os.getenv("SELECTED_GLOSS").split(",")
print(f"Gloss selezionati per l'indicizzazione: {target_glosses}")
SEQUENCE_LENGTH = 125
FEATURE_SIZE = 258
BATCH_SIZE = 500  

# Inizializzazione client Qdrant con retry
def get_qdrant_client(url, retries=10, delay=5):
    print(f"Tentativo di connessione a Qdrant su {url}...")
    for i in range(retries):
        try:
            client = QdrantClient(url=url)
            # Test connessione
            client.get_collections()
            print("Connesso a Qdrant con successo!")
            return client
        except Exception as e:
            print(f"Tentativo {i+1}/{retries} fallito: {e}. Riprovo tra {delay}s...")
            time.sleep(delay)
    raise Exception("Impossibile connettersi a Qdrant dopo diversi tentativi.")

client = get_qdrant_client(QDRANT_URL)

# load del modello
print(f"Caricamento modello da {MODEL_PATH}...")
encoder = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False
)



app = FastAPI(title="Sign Language Search API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_embedding(file_path):
    data = np.load(file_path)
    # Assicuriamoci che la dimensione sia corretta per il modello
    if data.shape[0] != SEQUENCE_LENGTH:
        data = resample_sequence(data)
    data = np.expand_dims(data, axis=0)
    embedding = encoder.predict(data, verbose=0)
    return embedding.flatten().tolist()

class KeypointsRequest(BaseModel):
    sequence: List[List[float]]  # [N][258]

def resample_sequence(sequence, target_frames=SEQUENCE_LENGTH):
    sequence = np.array(sequence)
    current_frames = len(sequence)
    
    if current_frames <= 1:
        return np.zeros((target_frames, FEATURE_SIZE))
    
    x_old = np.linspace(0, 1, current_frames)
    x_new = np.linspace(0, 1, target_frames)
    
    f = interpolate.interp1d(x_old, sequence, axis=0, kind='linear', fill_value="extrapolate")
    return f(x_new)

@app.post("/predict")
async def predict_gloss(request: KeypointsRequest):
    try:
        sequence = np.array(request.sequence, dtype=np.float32)
        
        if sequence.ndim != 2 or sequence.shape[1] != FEATURE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Shape errata: atteso (N, {FEATURE_SIZE}), ricevuto {sequence.shape}"
            )

        if len(sequence) != SEQUENCE_LENGTH:
            sequence = resample_sequence(sequence)
        
        sequence = np.expand_dims(sequence, axis=0)
        embedding = encoder.predict(sequence, verbose=0)[0]
        embedding_list = embedding.tolist()

        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding_list,
            limit=3
        )

        if not search_result.points:
            return {
                "predicted_gloss": "unknown",
                "confidence_score": 0.0,
                "message": "Nessun match trovato nel database"
            }

        best_match = search_result.points[0]
        
        # Debugging logging sicuro
        matches = [p.payload.get('gloss', 'N/A') for p in search_result.points]
        print(f"Top matches: {matches}")

        return {
            "predicted_gloss": best_match.payload.get("gloss", "unknown"),
            "confidence_score": best_match.score
        }

    except Exception as e:
        print(f"Errore durante la predizione: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

if __name__ == "__main__":
    print("\n--- INIZIALIZZAZIONE DATABASE ---")
    
    # Creazione/Reset collezione
    if client.collection_exists(COLLECTION_NAME):
        print(f"La collezione '{COLLECTION_NAME}' esiste già. La svuoto per ripopolarla...")
        client.delete_collection(collection_name=COLLECTION_NAME)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=256, distance=Distance.COSINE),
    )
    print(f"Collezione '{COLLECTION_NAME}' creata.")

    # Popolamento
    print(f"Cerco dati in: {os.path.abspath(DATA_PATH)}")
    if not os.path.exists(DATA_PATH):
        print(f"AVVISO: La cartella {DATA_PATH} non esiste. Il database rimarrà vuoto!")
    else:
        print(f"Contenuto di {DATA_PATH}: {os.listdir(DATA_PATH)[:10]}...")
        
        current_batch = []
        total_count = 0

        for gloss in tqdm(target_glosses, desc="Indicizzazione Gloss"):
            gloss_dir = os.path.join(DATA_PATH, gloss)
            if not os.path.isdir(gloss_dir):
                continue
            
            files = [f for f in os.listdir(gloss_dir) if f.endswith(".npy")]
            for file_name in files:
                file_path = os.path.join(gloss_dir, file_name)
                try:
                    vector = extract_embedding(file_path)
                    point = PointStruct(
                        id=str(uuid.uuid4()), 
                        vector=vector, 
                        payload={"gloss": gloss, "file": file_name}
                    )
                    current_batch.append(point)
                    
                    if len(current_batch) >= BATCH_SIZE:
                        client.upsert(collection_name=COLLECTION_NAME, points=current_batch)
                        total_count += len(current_batch)
                        current_batch = []
                except Exception as e:
                    print(f"Errore su {file_path}: {e}")

        if current_batch:
            client.upsert(collection_name=COLLECTION_NAME, points=current_batch)
            total_count += len(current_batch)

        print(f"Popolamento completato. Totale vettori: {total_count}")

    print("\n--- AVVIO SERVER API ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)
