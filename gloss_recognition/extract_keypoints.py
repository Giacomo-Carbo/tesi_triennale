import cv2
import mediapipe as mp
import numpy as np
import os
import json
import sys
from tqdm import tqdm
from scipy import interpolate
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

# --- CONFIGURAZIONE PATH ---
DATASET_PATH = "/Users/giacomocarbonara/.cache/kagglehub/datasets/risangbaskoro/wlasl-processed/versions/5"
VIDEOS_PATH = os.path.join(DATASET_PATH, "videos")
METADATA_PATH = os.path.join(DATASET_PATH, "WLASL_v0.3.json") 
OUTPUT_PATH = "MP_DATA_EMBEDDINGS"
TARGET_FRAMES = 125  # standardizzo a 125 frame per avere circa 5 secondi di video

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# uso media pipe per estrarre i keypoints e questa volta aggiungo anche una normalizzazione basata sulla distanza tra le spalle
# in questo modo, oltre a centrare tutto sul naso, rendo il modello immune al fatto che l'utente sia più o meno vicino alla camera
# se non rileva la posa, uso un centro di default per evitare che il modello riceva dati inconsistenti
def extract_keypoints(results):
    ref_x, ref_y = 0.5, 0.5
    scale_factor = 1.0

    if results.pose_landmarks:
        # il naso (indice 0) resta il mio punto di origine per la centratura
        ref_x = results.pose_landmarks.landmark[0].x
        ref_y = results.pose_landmarks.landmark[0].y
        
        # calcolo la distanza tra le spalle per capire quanto l'utente è vicino
        # dividendo le coordinate per questa distanza, i gesti avranno sempre la stessa proporzione
        shoulder_l = results.pose_landmarks.landmark[11]
        shoulder_r = results.pose_landmarks.landmark[12]
        dist = np.sqrt((shoulder_l.x - shoulder_r.x)**2 + (shoulder_l.y - shoulder_r.y)**2)
        if dist > 0:
            scale_factor = dist

    # estraggo i dati sottraendo il riferimento e dividendo per la scala
    # se mediapipe non trova nulla in quel frame, inserisco i soliti array di zeri per non rompere la sequenza
    pose = np.array([[(res.x - ref_x)/scale_factor, (res.y - ref_y)/scale_factor, res.z, res.visibility] 
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    lh = np.array([[(res.x - ref_x)/scale_factor, (res.y - ref_y)/scale_factor, res.z] 
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    rh = np.array([[(res.x - ref_x)/scale_factor, (res.y - ref_y)/scale_factor, res.z] 
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, lh, rh])

# è l'interpolazione che serve a uniformare il dataset: ogni video deve avere esattamente 125 frame
# rispetto al padding con zeri, questa tecnica è meglio perché "allunga" il gesto in modo fluido
# evitando che l'LSTM veda dei salti o dei momenti di vuoto totale alla fine del video
def resample_sequence(sequence, target_frames=TARGET_FRAMES):
    sequence = np.array(sequence)
    current_frames = len(sequence)
    
    if current_frames <= 1:
        return np.zeros((target_frames, 258))
    
    x_old = np.linspace(0, 1, current_frames)
    x_new = np.linspace(0, 1, target_frames)
    
    f = interpolate.interp1d(x_old, sequence, axis=0, kind='linear', fill_value="extrapolate")
    return f(x_new)

@contextmanager
def suppress_stdout_stderr():
    """Redirige stdout e stderr per evitare che i log di mediapipe intasino la console"""
    with open(os.devnull, 'w') as fnull:
        old_stdout_fd = os.dup(sys.stdout.fileno())
        old_stderr_fd = os.dup(sys.stderr.fileno())
        try:
            os.dup2(fnull.fileno(), sys.stdout.fileno())
            os.dup2(fnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            os.dup2(old_stdout_fd, sys.stdout.fileno())
            os.dup2(old_stderr_fd, sys.stderr.fileno())
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

# funzione che processa il video singolo, estrae i keypoints frame per frame e poi applica il resampling
# ho aggiunto un check sulla lunghezza minima (5 frame) per scartare video che potrebbero essere corrotti
# salvo il file in float32 così occupo meno spazio su disco mantenendo la precisione necessaria
def worker_process_video(video_info):
    video_path, output_file = video_info
    
    with suppress_stdout_stderr():
        mp_holistic = mp.solutions.holistic
        with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5) as model:
            cap = cv2.VideoCapture(video_path)
            sequence = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model.process(image)
                sequence.append(extract_keypoints(results))
                
            cap.release()
            
            if len(sequence) > 5:
                resampled_data = resample_sequence(sequence)
                np.save(output_file, resampled_data.astype(np.float32))
                return True
        return False

# --- MAIN ---

def main():
    print("Inizializzazione processo su M4 Pro...")
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    # raccolgo tutti i video che devono essere ancora processati
    all_tasks = []
    for entry in metadata:
        gloss = entry['gloss']
        gloss_dir = os.path.join(OUTPUT_PATH, gloss)
        if not os.path.exists(gloss_dir):
            os.makedirs(gloss_dir)
            
        for instance in entry['instances']:
            video_id = instance['video_id']
            video_path = os.path.join(VIDEOS_PATH, f"{video_id}.mp4")
            output_file = os.path.join(gloss_dir, f"{video_id}.npy")
            
            if os.path.exists(video_path) and not os.path.exists(output_file):
                all_tasks.append((video_path, output_file))

    print(f"Totale video da processare: {len(all_tasks)}")
    
    # uso un pool di 10 worker altrimenti in parallelo con mediapipe su M4 Pro rischia di saturare la CPU e rallentare tutto
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(worker_process_video, all_tasks), 
                  total=len(all_tasks), 
                  desc="Elaborazione Globale"))

    print("\nElaborazione completata!")

if __name__ == "__main__":
    main()