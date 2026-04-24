import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from mediapipe.framework.formats import landmark_pb2 
import time

# ==============================
# CONFIGURAZIONE
# ==============================
MODEL_PATH = Path("gesture_model") / "asl_gesture_classifier3.task"

# Disegnatori per i risultati
mp_drawing = mp.solutions.drawing_utils

# Definizione corretta delle connessioni (usiamo il modulo hands per le costanti)
mp_hands = mp.solutions.hands 
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS 

# ==============================
# FUNZIONE DI INFERENZA LIVE
# ==============================
def run_webcam_test():
    if not MODEL_PATH.exists():
        print(f"ERRORE: Modello non trovato in {MODEL_PATH}")
        return

    # --- 1. SETUP DEL RICONOSCITORE ---
    print(f"⏳ Caricamento modello da: {MODEL_PATH}")
    
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.4
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)
    print("✅ Modello caricato. Avvio Webcam...")

    # --- 2. SETUP WEBCAM ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERRORE: Impossibile accedere alla webcam.")
        return

    # --- 3. LOOP DI RICONOSCIMENTO ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 3. Esegui Riconoscimento
        recognition_result = recognizer.recognize(mp_image)

        # --- 4. Disegna Risultati ---
        
        # A) Disegna i landmark della mano
        if recognition_result.hand_landmarks:
            for hand_landmarks in recognition_result.hand_landmarks:
                
                # Conversione da lista di landmark a oggetto proto (necessario per draw_landmarks)
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                    for landmark in hand_landmarks
                ])
                
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks_proto, # Passiamo il proto object convertito
                    HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4), 
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2) 
                )

        # B) Scrivi la predizione
        if recognition_result.gestures:
            top_gesture = recognition_result.gestures[0][0]
            text = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
            color = (0, 255, 0) if top_gesture.score > 0.8 else (0, 165, 255)
        else:
            text = "Nessun Gesto Rilevato"
            color = (0, 0, 255)
            
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # 5. Mostra l'immagine
        cv2.imshow('ASL Gesture Recognizer (Live)', frame)

        # Esci premendo 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # --- 6. CLEANUP ---
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_webcam_test()