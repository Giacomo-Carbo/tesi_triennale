import os
import shutil
import cv2
import kagglehub
import tensorflow as tf
import mediapipe as mp
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from mediapipe_model_maker import gesture_recognizer
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import random
from PIL import Image
# ==============================
# CONFIGURAZIONE
# ==============================
DATASET_DIR = Path("dataset")
TESTSET_DIR = Path("testset")
MODEL_DIR = Path("gesture_model")
EXPORT_DIR = Path("exported_model")
MODEL_NAME = "asl_gesture_classifier3.task"

# Scaricamento dataset
DATASET_NUMBERS = Path(kagglehub.dataset_download("lexset/synthetic-asl-numbers")) / "Train_Nums"
DATASET_ALPAHBET = Path(kagglehub.dataset_download("grassknoted/asl-alphabet")) / "asl_alphabet_train/asl_alphabet_train"
TESTSET_NUMBERS_SRC = Path(kagglehub.dataset_download("lexset/synthetic-asl-numbers")) / "Test_Nums"
TESTSET_ALPAHBET_SRC = Path(kagglehub.dataset_download("grassknoted/asl-alphabet")) / "asl_alphabet_test/asl_alphabet_test"
print(TESTSET_NUMBERS_SRC)
MAX_IMAGES_PER_CLASS = 2000
EPOCH_NUM = 140
BATCH_SIZE = 128

# ==============================
# FUNZIONI DI UTILITÀ
# ==============================
def check_gpu():
    print("\n🔍 Verifica hardware...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU trovata: {gpus[0].name}")
    else:
        print("⚠️ GPU non trovata, uso della CPU.")
    print("-" * 30)

def merge_datasets(src_folders, dest_folder):
    dest_folder = Path(dest_folder)
    if dest_folder.exists() and any(dest_folder.iterdir()):
        print("✅ Dataset già unito. Salto il passaggio.")
        return

    dest_folder.mkdir(parents=True, exist_ok=True)
    for src in tqdm(src_folders, desc="Unione Dataset", unit="folder"):
        src_path = Path(src)
        if not src_path.exists(): continue
        list = [item for item in src_path.iterdir() if item.is_dir()]
        for item in tqdm(list, desc=f"Copiando da {src_path.name}", unit="img"):
            if item.is_dir():
                target_dir = dest_folder / item.name
                target_dir.mkdir(parents=True, exist_ok=True)
                for file in item.iterdir():
                    if file.is_file():
                        shutil.copy2(file, target_dir / file.name)

def validate_and_fix_dataset():
    print("🧹 Validazione e pulizia immagini...")
    none_dir = DATASET_DIR / "None"
    none_dir.mkdir(parents=True, exist_ok=True)

    # Unifica classi vuote in 'None' per MediaPipe
    for old_name in ["Nothing", "Blank", "nothing", "blank"]:
        old_dir = DATASET_DIR / old_name
        if old_dir.exists():
            for file in old_dir.iterdir():
                target = none_dir / file.name
                if not target.exists():
                    shutil.move(str(file), str(target))
                else:
                    os.remove(file)
            shutil.rmtree(old_dir)

    # Rimuove file corrotti e nascosti
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_files = [Path(r)/f for r, d, fs in os.walk(DATASET_DIR) for f in fs]
    
    for f in tqdm(all_files, desc="Verifica integrità", unit="img"):
        if f.name.startswith(".") or f.suffix.lower() not in valid_ext:
            os.remove(f)
            continue
        img = cv2.imread(str(f))
        if img is None:
            os.remove(f)

def prepare_testset():
    """Organizza il testset prendendo tutte le immagini dell'alfabeto e 1 per numero."""
    if TESTSET_DIR.exists(): return
    TESTSET_DIR.mkdir(parents=True, exist_ok=True)
    print("🧪 Preparazione testset...")

    # Alfabeto (tutte)
    for img in TESTSET_ALPAHBET_SRC.glob("*.jpg"):
        class_name = img.stem.split('_')[0]
        shutil.copy2(img, TESTSET_DIR / img.name)

    # Numeri (1 per cartella)
    for d in TESTSET_NUMBERS_SRC.iterdir():
        if d.is_dir():
            files = [f for f in d.iterdir() if f.is_file()]
            f = random.choice(files)
            shutil.copy2(f, TESTSET_DIR / (d.name+".png"))
            convert_png_to_jpg(TESTSET_DIR / (d.name+".png"), TESTSET_DIR / (d.name+".jpg"))
# ==============================
# CORE: TRAINING & INFERENZA
# ==============================
def load_data():
    print("📚 Caricamento dati...")
    data = gesture_recognizer.Dataset.from_folder(
        dirname=str(DATASET_DIR),
        hparams=gesture_recognizer.HandDataPreprocessingParams()
    )
    return data.split(0.9)

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

def test_model(model_path):
    print(f"\n🧪 Test inferenza su: {model_path}")
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)


    test_images = list(TESTSET_DIR.rglob("*.jpg"))
    for img_path in test_images:
        mp_image = mp.Image.create_from_file(str(img_path))
        res = recognizer.recognize(mp_image)
        label = res.gestures[0][0].category_name if res.gestures else "None"
        print(f"File: {img_path.name} -> Predizione: {label}")




def limit_images_per_folder(dataset_path, max_images):
    """
    Mantiene al massimo 'max_images' in ogni sottocartella, eliminando il resto.
    """
    dataset_path = Path(dataset_path)
    
    print(f"✂️ Ridimensionamento classi a max {max_images} immagini...")

    for category_dir in dataset_path.iterdir():
        if category_dir.is_dir():
            # Lista tutti i file immagine nella cartella
            images = [f for f in category_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
            
            if len(images) > max_images:
                # Mescola la lista per rimuovere file casuali
                random.shuffle(images)
                
                # Seleziona i file da eliminare
                to_delete = images[max_images:]
                
                for file in to_delete:
                    file.unlink() # Elimina il file
                
                print(f"✅ Classe {category_dir.name}: rimosse {len(to_delete)} immagini.")


def convert_png_to_jpg(input_path, output_path):
    # Apri l'immagine PNG
    img = Image.open(input_path)
    
    # Se l'immagine è in modalità RGBA (ha la trasparenza), 
    # dobbiamo convertirla in RGB prima di salvarla come JPG
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    
    # Salva come JPG con la qualità desiderata (da 1 a 100)
    img.save(output_path, "JPEG", quality=95)
    print(f"✅ Convertito: {output_path}")
# ==============================
# MAIN PIPELINE
# ==============================
if __name__ == "__main__":
    check_gpu()
    merge_datasets([DATASET_NUMBERS, DATASET_ALPAHBET], DATASET_DIR)
    limit_images_per_folder(DATASET_DIR, MAX_IMAGES_PER_CLASS)
    validate_and_fix_dataset()
    prepare_testset()
    
    train_data, val_data = load_data()
    model = train_model(train_data, val_data)
    
    # Esportazione
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.export_model(model_name=MODEL_NAME)
    final_path = MODEL_DIR / MODEL_NAME
    shutil.move(str(EXPORT_DIR / MODEL_NAME), str(final_path))
    
    print(f"💾 Modello finale salvato in: {final_path}")
    test_model(final_path)