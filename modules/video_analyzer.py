import cv2
import numpy as np

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    total = 0
    face_frames = 0
    positions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            face_frames += 1
            x, y, w, h = faces[0]
            positions.append(x)

    cap.release()

    face_presence = face_frames / max(total, 1)

    movement = np.std(positions) / 100 if positions else 0
    stability = 1 - movement

    return {
        "face_presence": face_presence,
        "movement": movement,
        "stability": stability
    }