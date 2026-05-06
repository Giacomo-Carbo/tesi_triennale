from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from pydantic import BaseModel
from typing import List
import numpy as np
import tensorflow as tf

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
MODEL_PATH = "final_lstm_encoder.h5"
SEQUENCE_LENGTH = 125
FEATURE_SIZE = 258

# --- LOAD MODEL ---
encoder = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("✅ Modello caricato")


# --- REQUEST MODEL ---
class KeypointsRequest(BaseModel):
    sequence: List[List[float]]  # [125][1662]


# --- ENDPOINT ---
@app.post("/predict")
async def predict_gloss(request: KeypointsRequest):
    try:
       
        sequence = np.array(request.sequence, dtype=np.float32)
        #contorllo sequenza
        if sequence.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
            raise HTTPException(
                status_code=400,
                detail=f"Shape errata: atteso ({SEQUENCE_LENGTH}, {FEATURE_SIZE}), ricevuto {sequence.shape}"
            )

       
        sequence = np.expand_dims(sequence, axis=0)  # (1,125,258)
        embedding = encoder.predict(sequence, verbose=0)[0]
        print(f"Embedding generato: {embedding.shape}")
        embedding_list = embedding.tolist()

       
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding_list,
            limit=1
        )

        print(f"se non va mi sparo: {search_result}")
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