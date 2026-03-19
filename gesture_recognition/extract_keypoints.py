import kagglehub
import cv2
import mediapipe as mp
import numpy as np
import os
import json
from tqdm import tqdm
from scipy import interpolate
from concurrent.futures import ProcessPoolExecutor

# --- CONFIGURAZIONE PERCORSI ---
DATASET_PATH = kagglehub.dataset_download("risangbaskoro/wlasl-processed")
VIDEOS_PATH = os.path.join(DATASET_PATH, "videos")
METADATA_PATH = os.path.join(DATASET_PATH, "WLASL_v0.3.json") 
OUTPUT_PATH = "MP_DATA_EMBEDDINGS"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# --- FUNZIONI DI SUPPORTO (Invariate) ---

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def resample_sequence(sequence, target_frames=60):
    current_frames = len(sequence)
    if current_frames <= 1: # Protezione per sequenze troppo corte
        return np.zeros((target_frames, 1662))
    
    sequence = np.array(sequence)
    x_old = np.linspace(0, 1, current_frames)
    x_new = np.linspace(0, 1, target_frames)
    f = interpolate.interp1d(x_old, sequence, axis=0, kind='linear', fill_value="extrapolate")
    return f(x_new)

# --- WORKER PER IL MULTIPROCESSING ---

def worker_process_video(video_info):
    video_path, output_file, target_length = video_info
    
    mp_holistic = mp.solutions.holistic
    # Ridotto l'overhead di MediaPipe per processi paralleli
    with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5) as model:
        cap = cv2.VideoCapture(video_path)
        sequence = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = model.process(image)
            sequence.append(extract_keypoints(results))
            
        cap.release()
        
        if len(sequence) > 0:
            resampled_data = resample_sequence(sequence, target_frames=target_length)
            np.save(output_file, resampled_data)
            return True
    return False

# --- MAIN ---

def main():
    if not os.path.exists(METADATA_PATH):
        print(f"Errore: Metadati non trovati.")
        return

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    # Iteriamo parola per parola (SEQUENZIALE)
    for entry in tqdm(metadata, desc="Avanzamento Parole"):
        gloss = entry['gloss']
        gloss_dir = os.path.join(OUTPUT_PATH, gloss)
        
        if not os.path.exists(gloss_dir):
            os.makedirs(gloss_dir)
            
        # Prepariamo i task solo per i video di questa specifica parola
        word_tasks = []
        for instance in entry['instances']:
            video_id = instance['video_id']
            video_path = os.path.join(VIDEOS_PATH, f"{video_id}.mp4")
            output_file = os.path.join(gloss_dir, f"{video_id}.npy")
            
            if os.path.exists(video_path) and not os.path.exists(output_file):
                word_tasks.append((video_path, output_file, 60))

        # Se ci sono video da processare per questa parola, usiamo il parallelismo
        if word_tasks:
            # Il pool si apre e si chiude per ogni parola
            with ProcessPoolExecutor() as executor:
                # Esegue i video della parola corrente in parallelo sui core CPU
                list(executor.map(worker_process_video, word_tasks))

    print("\nElaborazione completata!")

if __name__ == "__main__":
    main()