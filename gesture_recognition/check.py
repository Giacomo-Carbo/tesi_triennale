import mediapipe as mp
try:
    holistic = mp.solutions.holistic.Holistic()
    print("MediaPipe funziona.")
    holistic.close()
except Exception as e:
    print(f"Errore: {e}")

