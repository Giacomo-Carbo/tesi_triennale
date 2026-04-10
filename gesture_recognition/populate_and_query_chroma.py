import os
import numpy as np
import tensorflow as tf
import chromadb
from chromadb.config import Settings

# --- 1. CONFIGURAZIONE ---
MODEL_PATH = "final_lstm_encoder.h5"
DATA_PATH = "MP_DATA_EMBEDDINGS" # La cartella con le sottocartelle delle glosse
CHROMA_DB_DIR = "asl_vector_db"

# Carica l'encoder addestrato
# Nota: safe_mode=False o custom_objects potrebbero servire se hai usato la Triplet Loss
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

# --- 3. POPOLAMENTO DEL DATABASE ---
print("📂 Inizio indicizzazione dei segni in ChromaDB...")

all_embeddings = []
all_metadatas = []
all_ids = []

# Ciclo su tutte le cartelle (glosse) e i file
for gloss in os.listdir(DATA_PATH):
    gloss_dir = os.path.join(DATA_PATH, gloss)
    if not os.path.isdir(gloss_dir): continue
    
    print(f"Processando gloss: {gloss}")
    for i, file_name in enumerate(os.listdir(gloss_dir)):
        if not file_name.endswith(".npy"): continue
        
        file_path = os.path.join(gloss_dir, file_name)
        
        # Genera il vettore
        vector = extract_embedding(file_path)
        
        all_embeddings.append(vector)
        all_metadatas.append({"gloss": gloss, "file": file_name})
        all_ids.append(f"{gloss}_{i}")

# Caricamento massivo in ChromaDB
collection.add(
    embeddings=all_embeddings,
    metadatas=all_metadatas,
    ids=all_ids
)

print(f"✅ Database pronto con {len(all_ids)} vettori!")

# --- 4. ESEMPIO DI QUERY (RICERCA) ---
print("\n🔍 Esempio di ricerca semantica...")

# Immaginiamo di avere un nuovo video "misterioso" (qui ne prendiamo uno a caso dal dataset)
test_gloss = random.choice(os.listdir(DATA_PATH))
test_file = os.listdir(os.path.join(DATA_PATH, test_gloss))[0]
test_path = os.path.join(DATA_PATH, test_gloss, test_file)

query_vector = extract_embedding(test_path)

# Cerchiamo i 3 segni più vicini nello spazio vettoriale
results = collection.query(
    query_embeddings=[query_vector],
    n_results=3
)

print(f"\nRisultati per il segno di test '{test_gloss}':")
for i in range(len(results['ids'][0])):
    print(f"{i+1}. Glossa: {results['metadatas'][0][i]['gloss']} (ID: {results['ids'][0][i]}) - Distanza: {results['distances'][0][i]}")