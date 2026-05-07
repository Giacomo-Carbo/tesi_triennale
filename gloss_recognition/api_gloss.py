from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from pydantic import BaseModel
from typing import List
import numpy as np
import tensorflow as tf
from scipy import interpolate
# --- CONFIGURAZIONE ---
app = FastAPI(title="Sign Language Search API")

# CORS (per JS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "gloss"
MODEL_PATH = "final_lstm_encoder.keras"
SEQUENCE_LENGTH = 125
FEATURE_SIZE = 258

# --- LOAD MODEL ---
encoder = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False
)

print("✅ Modello caricato")


# --- REQUEST MODEL ---
class KeypointsRequest(BaseModel):
    sequence: List[List[float]]  # [125][258]

def resample_sequence(sequence, target_frames=SEQUENCE_LENGTH):
    sequence = np.array(sequence)
    current_frames = len(sequence)
    
    if current_frames <= 1:
        return np.zeros((target_frames, 258))
    
    x_old = np.linspace(0, 1, current_frames)
    x_new = np.linspace(0, 1, target_frames)
    
    f = interpolate.interp1d(x_old, sequence, axis=0, kind='linear', fill_value="extrapolate")
    return f(x_new)

# --- ENDPOINT ---
@app.post("/predict")
async def predict_gloss(request: KeypointsRequest):
    try:
       
        sequence = np.array(request.sequence, dtype=np.float32)
        
        # Controllo che la dimensione delle feature sia corretta (258)
        # Permettiamo invece una lunghezza della sequenza variabile (N frame)
        if sequence.ndim != 2 or sequence.shape[1] != FEATURE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Shape errata: atteso (N, {FEATURE_SIZE}), ricevuto {sequence.shape}"
            )

        # Resampling della sequenza se diversa da SEQUENCE_LENGTH
        if len(sequence) != SEQUENCE_LENGTH:
            print(f"Resampling sequenza da {len(sequence)} a {SEQUENCE_LENGTH} frame")
            sequence = resample_sequence(sequence)
        
        sequence = np.expand_dims(sequence, axis=0)  # (1,125,258)
        
        embedding = encoder.predict(sequence, verbose=0)[0]
        print(f"Embedding generato: {embedding.shape}")
        embedding_list = embedding.tolist()

       
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding_list,
            limit=3
        )

        print(f"se non va mi sparo: {search_result.points[0].payload['gloss']}, {search_result.points[1].payload['gloss']}, {search_result.points[2].payload['gloss']}")
        best_match = search_result.points[0]
        return {
            "predicted_gloss": best_match.payload["gloss"],
            "confidence_score": best_match.score
        }

    except Exception as e:
        print(f"Errore durante la predizione: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --- RUN ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)