import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose

cap = cv2.VideoCapture(0)
# Forza la finestra
cv2.namedWindow('Pose', cv2.WINDOW_NORMAL)

# Debug: verifica apertura
if not cap.isOpened():
    print("ERRORE: la webcam non si apre (VideoCapture(0) fallito).")
    exit()
else:
    print("Webcam aperta correttamente. Premendo 'q' chiuderai.")

# Prova un singolo frame di test
ret, frame = cap.read()
if not ret:
    print("ERRORE: impossibile leggere un frame dalla webcam.")
    cap.release()
    exit()
else:
    print("Frame di test letto, mostro per 3s…")
    cv2.imshow('Pose', frame)
    cv2.waitKey(3000)  # 3 secondi
    cv2.destroyWindow('Pose')

# Ora il loop principale
with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame non ricevuto, esco dal loop.")
            break

        # elaborazione
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Pose', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Hai premuto q, esco.")
            break
        elif key != 255:
            # Qualsiasi altro tasto premuto
            print(f"Tasto premuto: {chr(key)} ({key})")

cap.release()
cv2.destroyAllWindows()
