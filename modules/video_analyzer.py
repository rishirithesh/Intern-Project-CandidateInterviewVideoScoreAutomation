import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple

from config import Config


class VideoAnalysisResult:
    def __init__(
        self,
        face_presence: float,
        eye_contact: float,
        lighting: float,
        background_cleanliness: float,
        camera_stability: float,
        grooming: float,
        dressing: float,
        professionalism: float,
        face_confidence: float,
    ):
        self.face_presence = face_presence
        self.eye_contact = eye_contact
        self.lighting = lighting
        self.background_cleanliness = background_cleanliness
        self.camera_stability = camera_stability
        self.grooming = grooming
        self.dressing = dressing
        self.professionalism = professionalism
        self.face_confidence = face_confidence


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _normalize(value: float, minimum: float, maximum: float) -> float:
    return _clamp((value - minimum) / max(maximum - minimum, 1e-9))


def _skin_mask_ratio(frame: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 10, 60], dtype=np.uint8)
    upper = np.array([25, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return float(np.count_nonzero(mask) / max(mask.size, 1))


def _extract_face_region(frame: np.ndarray, landmarks) -> Optional[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    if frame is None:
        return None
    h, w, _ = frame.shape
    xs = [int(p.x * w) for p in landmarks]
    ys = [int(p.y * h) for p in landmarks]
    x1, x2 = max(0, min(xs)), min(w, max(xs))
    y1, y2 = max(0, min(ys)), min(h, max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1), frame[y1:y2, x1:x2]


def _sample_timestamps(duration: float, count: int) -> list[float]:
    if duration <= 0 or count < 1:
        return [0.0]
    if count == 1:
        return [0.0]
    return [min(duration - 0.01, i * duration / (count - 1)) for i in range(count)]


def analyze_video(video_path: str) -> VideoAnalysisResult:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError('Unable to open video for analysis')

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    sample_count = min(Config.VIDEO_SAMPLE_COUNT, max(1, int(duration / Config.VIDEO_MIN_SAMPLE_SECONDS)))
    timestamps = _sample_timestamps(duration, sample_count)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    face_detections = 0
    brightness_values = []
    edge_density_values = []
    centers = []
    eye_contact_values = []
    clothing_scores = []
    grooming_values = []
    frame_height = 0
    frame_width = 0

    for timestamp in timestamps:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ret, frame = capture.read()
        if not ret or frame is None:
            continue

        frame_height, frame_width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(np.mean(gray) / 255.0)
        edges = cv2.Canny(gray, 80, 180)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            edge_density_values.append(np.count_nonzero(edges) / max(edges.size, 1))
            continue

        face_detections += 1
        landmarks = results.multi_face_landmarks[0].landmark
        face_region_data = _extract_face_region(frame, landmarks)
        if face_region_data is None:
            edge_density_values.append(np.count_nonzero(edges) / max(edges.size, 1))
            continue

        box, face_roi = face_region_data
        x, y, w_box, h_box = box
        centers.append((x + w_box / 2.0, y + h_box / 2.0))

        left_eye = np.mean([[landmarks[i].x, landmarks[i].y] for i in (33, 133)], axis=0)
        right_eye = np.mean([[landmarks[i].x, landmarks[i].y] for i in (362, 263)], axis=0)
        horizontal_offset = abs((left_eye[0] + right_eye[0]) / 2.0 - 0.5)
        eye_contact_values.append(_clamp(1.0 - horizontal_offset * 3.5))

        face_brightness = np.mean(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)) / 255.0
        face_visibility = _clamp((w_box * h_box) / max(frame_height * frame_width, 1) * 10.0)
        grooming_values.append(_clamp((face_brightness - 0.25) * 1.9) * face_visibility)

        shoulder_bottom = min(frame_height, int(y + h_box * 2.2))
        x_end = min(frame_width, x + w_box)
        clothing_region = frame[int(y + h_box * 0.9):shoulder_bottom, x:x_end]
        if clothing_region is not None and clothing_region.size > 0:
            skin_ratio = _skin_mask_ratio(clothing_region)
            clothing_scores.append(_clamp(1.0 - skin_ratio * 1.5))
        else:
            clothing_scores.append(0.5)

        mask = np.ones(edges.shape, dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x + w_box, y + h_box), 0, thickness=-1)
        edge_density_values.append(np.count_nonzero(edges[mask == 1]) / max(np.count_nonzero(mask == 1), 1))

    capture.release()
    face_mesh.close()

    frames_sampled = max(1, len(timestamps))
    face_presence = _clamp(face_detections / frames_sampled)

    camera_stability = 0.6
    if len(centers) > 1 and frame_width > 0:
        motions = [np.linalg.norm(np.array(centers[i]) - np.array(centers[i - 1])) for i in range(1, len(centers))]
        average_motion = float(np.mean(motions))
        camera_stability = _clamp(1.0 - (average_motion / frame_width) * 6.0)

    lighting = _clamp((float(np.mean(brightness_values)) - 0.3) / 0.5) if brightness_values else 0.35
    background_cleanliness = _clamp(1.0 - float(np.mean(edge_density_values)) * 1.7) if edge_density_values else 0.45
    average_eye_contact = float(np.mean(eye_contact_values)) if eye_contact_values else 0.35
    average_grooming = float(np.mean(grooming_values)) if grooming_values else 0.45
    average_dressing = float(np.mean(clothing_scores)) if clothing_scores else 0.5
    professionalism = _clamp(
        0.22 * face_presence +
        0.22 * lighting +
        0.2 * average_eye_contact +
        0.2 * background_cleanliness +
        0.16 * camera_stability
    )

    return VideoAnalysisResult(
        face_presence=round(face_presence, 3),
        eye_contact=round(average_eye_contact, 3),
        lighting=round(lighting, 3),
        background_cleanliness=round(background_cleanliness, 3),
        camera_stability=round(camera_stability, 3),
        grooming=round(average_grooming, 3),
        dressing=round(average_dressing, 3),
        professionalism=round(professionalism, 3),
        face_confidence=round(_clamp(face_detections / frames_sampled), 3),
    )
