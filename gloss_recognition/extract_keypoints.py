import kagglehub
import cv2
import mediapipe as mp
import numpy as np
import os
import json
import sys
from tqdm import tqdm
from scipy import interpolate
from concurrent.futures import ProcessPoolExecutor


#DATASET_PATH = kagglehub.dataset_download("risangbaskoro/wlasl-processed")
DATASET_PATH = "/Users/giacomocarbonara/.cache/kagglehub/datasets/risangbaskoro/wlasl-processed/versions/5"
VIDEOS_PATH = os.path.join(DATASET_PATH, "videos")
METADATA_PATH = os.path.join(DATASET_PATH, "WLASL_v0.3.json") 
OUTPUT_PATH = "MP_DATA_EMBEDDINGS"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

#uso media pipe per estrarre i keypoints da ogni video, e li concateno in un singolo array 
#in oltre se in quel determinato frame non vengono rilevati keypoints, inserisco un array di zeri
#la funzione extract_keypoints estrae i keypoints da un singolo frame, centrando le coordinate rispetto al naso (se rilevato)
#in modo da rendere il modello più robusto a variazioni di posizione e distanza dell'utente rispetto alla camera
def extract_keypoints(results):
    # 1. Estraiamo il riferimento (Naso) se disponibile
    # Il naso è il landmark 0 nella posa.
    if results.pose_landmarks:
        ref_x = results.pose_landmarks.landmark[0].x
        ref_y = results.pose_landmarks.landmark[0].y
    else:
        ref_x, ref_y = 0, 0

    # 2. Estrazione con sottrazione del riferimento (Centratura)
    # Sottraiamo ref_x dalle coordinate x (indice 0) e ref_y dalle coordinate y (indice 1)
    
    pose = np.array([[res.x - ref_x, res.y - ref_y, res.z, res.visibility] 
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    face = np.array([[res.x - ref_x, res.y - ref_y, res.z] 
                     for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    
    lh = np.array([[res.x - ref_x, res.y - ref_y, res.z] 
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    rh = np.array([[res.x - ref_x, res.y - ref_y, res.z] 
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, face, lh, rh])


#è essenzialmente un interpolazione che viene fatta sui keypoints estratti da media pipe, 
#in modo da avere sempre 100 frame per ogni video (5 secondi a 25 FPS)
def resample_sequence(sequence, target_frames=125):
    current_frames = len(sequence)
    if current_frames <= 1: # Protezione per sequenze troppo corte
        return np.zeros((target_frames, 1662))
    sequence = np.array(sequence)
    x_old = np.linspace(0, 1, current_frames)
    x_new = np.linspace(0, 1, target_frames)
    f = interpolate.interp1d(x_old, sequence, axis=0, kind='linear', fill_value="extrapolate")
    return f(x_new)

#necessaria pk dato che nella piattaforma di learning dara una specie di finiestra 5 sec in cui l'utente potrà fare il
#gesto se esso dura dimeno l'utnte può bloccare e riempirò di zeri i frame non "utilizzati"
#quindi se una sequenza è più corta di 100 frame, la riempio con zeri fino a raggiungere i 100 frame richiesti
#pk 4 sec? perche il gesto più lungo dura 4 secondi
def uniform_lenght(sequence, target_frames=100):
    sequence = np.array(sequence)
    
    # Gestione dimensioni per evitare il ValueError precedente
    if sequence.ndim == 1:
        sequence = sequence.reshape(-1, 1662) if sequence.size > 0 else np.empty((0, 1662))
            
    current_frames = sequence.shape[0]
    
    # Se il video è troppo lungo, lo scartiamo restituendo None
    if current_frames > target_frames:
        return None

    # Se è più corto, facciamo padding con zeri
    padding = np.zeros((target_frames - current_frames, 1662))
    return np.concatenate([sequence, padding], axis=0)


#funzione che processa un singolo video, estrae i keypoints e salva il risultato in un file .npy
#ho aggironato la funzione per usare il multiporcessing, in modo da processare più video in parallelo e velocizzare l'elaborazione 
def worker_process_video(video_info):
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    video_path, output_file, target_length = video_info
    
    mp_holistic = mp.solutions.holistic
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
        sequence = uniform_lenght(sequence)
        if (sequence is not None) and (len(sequence) > 0):
            #resampled_data = resample_sequence(sequence, target_frames=target_length)
            #np.save(output_file, resampled_data)
            np.save(output_file, sequence) #salvo la sequenza originale senza resampling
            return True
    return False

# --- MAIN ---

def main():
    print("Inizializzazione processo...")
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    # 1. Raccogliamo TUTTI i task di tutti i video prima di iniziare
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
                all_tasks.append((video_path, output_file, 100))

    # 2. Avviamo un UNICO Pool di processi per tutto il dataset
    # Su M4 Pro posso spingermi ad usare un numero di worker pari ai core (es. 10 o 12)
    print(f"Totale video da processare: {len(all_tasks)}")
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Usiamo tqdm per vedere l'avanzamento globale reale
        list(tqdm(executor.map(worker_process_video, all_tasks), 
                  total=len(all_tasks), 
                  desc="Elaborazione Globale"))

    print("\nElaborazione completata!")

if __name__ == "__main__":
    main()