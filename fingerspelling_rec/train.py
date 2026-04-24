# Per utilizzare, creare un ambiente virtuale con Python 3.11
# con i seguenti comandi:
#
# python3.11 -m venv .venv
# source venv/bin/activate
# pip install -r requirements.txt
# pip install -r requirements_model_maker.txt --no-deps

import os
from pathlib import Path
import kagglehub
import cv2
import mediapipe as mp
from mediapipe_model_maker import gesture_recognizer
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import shutil

# ==============================
# CONFIGURAZIONE
# ==============================
DATASET_NUMBERS = Path(kagglehub.dataset_download("lexset/synthetic-asl-numbers")) / "Train_Nums"
print(f"Dataset numeri in: {DATASET_NUMBERS}")
DATASET_ALPAHBET = Path(kagglehub.dataset_download("grassknoted/asl-alphabet")) / "asl_alphabet_train/asl_alphabet_train"
print(f"Dataset alfabeto in: {DATASET_ALPAHBET}")
DATASET_DIR = Path("dataset")
TESTSET_NUMBERS = Path(kagglehub.dataset_download("lexset/synthetic-asl-numbers")) / "Test_Nums"
TESTSET_ALPAHBET = Path(kagglehub.dataset_download("grassknoted/asl-alphabet")) / "asl_alphabet_test/asl_alphabet_test"

MODEL_DIR = Path("gesture_model")
MODEL_NAME = "asl_gesture_classifier3.task"
EPOCH_NUM = 80

# ==============================
# FUNZIONI DI UTILITÀ
# ==============================
def check_gpu():
    """Controlla la presenza di GPU e stampa lo stato."""
    print("Verifica della configurazione hardware...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("✅ GPU trovata. TensorFlow la utilizzerà per l'addestramento.")
        for gpu in gpus:
            print(f"  - Dispositivo: {gpu.name}")
    else:
        print("⚠️ Nessuna GPU trovata. TensorFlow utilizzerà la CPU.")
        print("Per l'accelerazione su Mac, installa 'tensorflow-metal' con: pip install tensorflow-metal")
    print("-" * 30)




# ==============================
# CORREZIONE E VALIDAZIONE DEL DATASET
# ==============================
def validate_and_fix_dataset():
    """
    Valida la struttura e l'integrità del dataset.
    - Rinomina la directory 'Nothing' e 'Blank' in 'None'.
    - Rimuove i file nascosti (come .DS_Store).
    - Controlla e rimuove le immagini corrotte.
    """

    print("🧹 Pulizia e verifica delle immagini...")
    
    # Definizione percorsi
    nothing = DATASET_DIR / "Nothing"
    blank = DATASET_DIR / "Blank"
    none = DATASET_DIR / "none"

    # CREAZIONE CARTELLA TARGET (Cruciale per evitare FileNotFoundError)
    none.mkdir(exist_ok=True)

    # Funzione interna per evitare ripetizioni di codice
    def merge_into_none(src_dir):
        if src_dir.exists() and src_dir.is_dir():
            print(f"Spostamento file da {src_dir.name} a None...")
            for file in src_dir.iterdir():
                if file.is_file():
                    target_path = none / file.name
                    if target_path.exists():
                        # Se il file esiste già in 'None', lo eliminiamo dalla sorgente
                        os.remove(file)
                    else:
                        shutil.move(str(file), str(target_path))
            # Rimuove la cartella sorgente ormai vuota
            shutil.rmtree(src_dir)

    # Esecuzione unione cartelle
    merge_into_none(nothing)
    merge_into_none(blank)

    # Pulizia file nascosti e raccolta immagini
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    files_to_check = []

    for root, dirs, files in os.walk(DATASET_DIR):
        for f in files:
            file_path = Path(root) / f
            # Rimuove file nascosti (es. .DS_Store)
            if f.startswith("."):
                os.remove(file_path)
            elif file_path.suffix.lower() in image_extensions:
                files_to_check.append(file_path)

    # Verifica integrità immagini
    valid_count = 0
    removed_count = 0

    print(f"🔍 Analisi di {len(files_to_check)} immagini...")
    for file_path in tqdm(files_to_check, desc="Validazione Dataset", unit="img"):
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                os.remove(file_path)
                removed_count += 1
            else:
                valid_count += 1
        except Exception as e:
            print(f"\nErrore con {file_path.name}: {e}. Rimosso.")
            if file_path.exists():
                os.remove(file_path)
            removed_count += 1

    print(f"\n✅ Validazione completata: {valid_count} immagini valide, {removed_count} rimosse.\n")




# ==============================
# CARICAMENTO DEL DATASET
# ==============================

def load_dataset():
    """
    Carica il dataset utilizzando MediaPipe Model Maker e lo divide in set di addestramento, validazione e test
    """
    print("📚 Caricamento dati in MediaPipe Model Maker...")
    # Nota: from_folder non supporta una barra di avanzamento, ma abbiamo già convalidato i file.
    data = gesture_recognizer.Dataset.from_folder(
        dirname=str(DATASET_DIR),
        hparams=gesture_recognizer.HandDataPreprocessingParams()
    )

    train_data, validation_test_data = data.split(0.8)
    validation_data, test_data = validation_test_data.split(0.5)
    print(f"📊 Dati caricati. Addestramento: {len(train_data)}, Validazione: {len(validation_data)}")

    return train_data, validation_data


# ==============================
# ADDESTRAMENTO DEL MODELLO
# ==============================

def train_model(train_data, validation_data):
    """
    Addestra il modello di riconoscimento dei gesti
    """
    print("🏋️‍♂️ Inizio dell'addestramento del modello...")

    hparams = gesture_recognizer.HParams(
        epochs=EPOCH_NUM,
        batch_size=128,
        learning_rate=0.005,
        export_dir="exported_model"
    )

    model_options = gesture_recognizer.ModelOptions(dropout_rate=0.05)

    options = gesture_recognizer.GestureRecognizerOptions(
        model_options=model_options,
        hparams=hparams
    )

    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options
    )

    print("🏁 Addestramento completato!")
    return model


# ==============================
# ESPORTAZIONE DEL MODELLO
# ==============================

def save_model(model):
    """
    Esporta il modello
    """
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / MODEL_NAME

    # export_model salva anche i metadati necessari.
    model.export_model(model_name=MODEL_NAME)

    # Sposta il file nella nostra cartella desiderata per coerenza.
    source_path = Path("exported_model") / MODEL_NAME
    if source_path.exists():
        shutil.move(str(source_path), str(model_path))

    print(f"💾 Modello salvato in: {model_path}")
    return model_path


# ==============================
# TEST DEL MODELLO (INFERENZA)
# ==============================

def test_model(model_path):
    """
    Testa il modello esportato
    """
    print("\n🧪 Inizio del test del modello con l'API MediaPipe Tasks...")

    # Configurazione di BaseOptions per l'inferenza
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.GestureRecognizerOptions(base_options=base_options)

    # Crea il riconoscitore
    recognizer = vision.GestureRecognizer.create_from_options(options)

    test_files = [f for f in TESTSET_DIR.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]

    print(f"Test su {len(test_files)} immagini di test...")

    for image_path in test_files:
        # Carica l'immagine come oggetto MediaPipe Image
        mp_image = mp.Image.create_from_file(str(image_path))

        # Esegui il riconoscimento
        recognition_result = recognizer.recognize(mp_image)

        top_gesture = "None"
        score = 0.0

        if recognition_result.gestures:
            top_gesture = recognition_result.gestures[0][0].category_name
            score = recognition_result.gestures[0][0].score

        print(f"File: {image_path.name} -> Predizione: {top_gesture} (Punteggio: {score:.2f})")







#==============================
# UNIONE DEI DATASET
#==============================
def merge_datasets(src_folders, dest_folder):
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)

    
    for src in tqdm(src_folders, desc="Elaborazione sorgenti", unit="folder"):
        src_path = Path(src)
        
        if not src_path.exists():
            print(f"Avviso: {src} non trovato.")
            continue

        # Scorre tutte le sottocartelle (classi A, B, C...)
        subdirs = [d for d in src_path.iterdir() if d.is_dir()]
        
        for item in subdirs:
            target_dir = dest_folder / item.name
            target_dir.mkdir(exist_ok=True)
            
            # Lista dei file per conoscere il totale e alimentare tqdm
            files = [f for f in item.iterdir() if f.is_file()]
            
            # Barra interna per il monitoraggio dei singoli file
            desc_file = f"Copia classe {item.name}"
            for file in tqdm(files, desc=desc_file, unit="file", leave=False):
                shutil.copy2(file, target_dir / file.name)
    
    print(f"\nUnione completata in: {dest_folder}")



# ==============================
# PIPELINE PRINCIPALE
# ==============================

if __name__ == "__main__":
    print("\nAvvio\n")
    check_gpu()
    merge_datasets([DATASET_NUMBERS, DATASET_ALPAHBET], DATASET_DIR)
    merge_datasets([TESTSET_NUMBERS, TESTSET_ALPAHBET], TESTSET_DIR)
    validate_and_fix_dataset()
    train_data, validation_data = load_dataset()
    model = train_model(train_data, validation_data)
    #loss, acc = model.evaluate(test_data, batch_size=1)
    #print(f"\n\n\n\n\nLoss sul test set: {loss}, Accuratezza sul test set: {acc}\n\n\n\n\n\n")

    saved_path = save_model(model)

    # Test di inferenza reale su file esterni
    if TESTSET_DIR.exists():
       test_model(saved_path)
    else:
      print("⚠️ Cartella di test non trovata, salto dell'inferenza.")

    print("\nsuccesso!\n")