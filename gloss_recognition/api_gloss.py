from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient
from pydantic import BaseModel
from typing import List

# --- CONFIGURAZIONE ---
app = FastAPI(title="Sign Language Search API")
client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "gloss"

# Modello per ricevere il vettore di input
class SearchRequest(BaseModel):
    embedding: List[float]

@app.post("/glossquery")
async def find_nearest_gloss(request: SearchRequest):
    """
    Riceve un embedding di dimensione 256 e restituisce la gloss più vicina.
    """
    try:
        # 1. Esegui la ricerca nel DB Vettoriale
        # Cerchiamo il più vicino (limit=1)
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=request.embedding,
            limit=1,
            with_payload=True  # Fondamentale per leggere il campo "gloss"
        )

        # 2. Controllo se abbiamo trovato qualcosa
        if not search_result:
            raise HTTPException(status_code=404, detail="Nessuna gloss trovata per questo vettore")

        # 3. Estrai i dati dal risultato più vicino
        best_match = search_result.points[0]
        gloss_name = best_match.payload.get("gloss")
        score = best_match.score

        return {
            "predicted_gloss": gloss_name,
            "confidence_score": score,
            "metadata": best_match.payload
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)