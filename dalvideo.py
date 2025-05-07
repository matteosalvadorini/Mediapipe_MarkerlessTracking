import os
# Debug: mostra directory di lavoro e verifica file
cwd = os.getcwd()
print(f"Current working directory: {cwd}")

video_path = 'video.mp4'  # <--- assicurati che il file sia qui o inserisci il path corretto
print(f"Video path: {video_path}")
print(f"File exists: {os.path.exists(video_path)}")

# Silenzia log di TensorFlow e absl
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import cv2
import mediapipe as mp

# Inizializza MediaPipe Pose
drawing_utils = mp.solutions.drawing_utils
pose_module = mp.solutions.pose

# Apri il video da file
cap = cv2.VideoCapture(video_path)
# Crea finestra ridimensionabile
cv2.namedWindow('Pose', cv2.WINDOW_NORMAL)

# Verifica apertura
if not cap.isOpened():
    print(f"ERRORE: impossibile aprire il file video '{video_path}'")
    print("Controlla il path o il nome del file.")
    exit()
else:
    print(f"Video '{video_path}' aperto correttamente. Premi Invio per avviare il loop..." )
    input()

# Calcola FPS e delay per rallentamento
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
slow_factor = 4.0
# Millisecondi di attesa tra frame
delay = int(1000.0 / fps * slow_factor)
print(f"FPS video: {fps:.2f}, delay per frame: {delay} ms (rallentamento {slow_factor}×)")

# Indici dei landmarks da "riparare" in caso di occlusione
landmark_indices = [
    pose_module.PoseLandmark.LEFT_HIP,
    pose_module.PoseLandmark.LEFT_KNEE,
    pose_module.PoseLandmark.LEFT_ANKLE,
    pose_module.PoseLandmark.RIGHT_HIP,
    pose_module.PoseLandmark.RIGHT_KNEE,
    pose_module.PoseLandmark.RIGHT_ANKLE
]

prev_landmarks = None

# Loop di elaborazione
with pose_module.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        print(f"Read frame: ret={ret}")
        if not ret:
            print("Fine del video o frame non ricevuto, esco.")
            break

        # Prepara immagine per MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Ripara occlusioni
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            if prev_landmarks is not None:
                for idx in landmark_indices:
                    if lm[idx].visibility < 0.1:
                        lm[idx].x = prev_landmarks[idx].x
                        lm[idx].y = prev_landmarks[idx].y
                        lm[idx].z = prev_landmarks[idx].z
                        lm[idx].visibility = prev_landmarks[idx].visibility
            prev_landmarks = [l for l in lm]
            drawing_utils.draw_landmarks(image, results.pose_landmarks, pose_module.POSE_CONNECTIONS)

        # Mostra frame
        cv2.imshow('Pose', image)

        key = cv2.waitKey(delay) & 0xFF
        print(f"waitKey returned: {key}")
        if key == ord('q'):
            print("Hai premuto 'q', esco.")
            break

cap.release()
cv2.destroyAllWindows()


