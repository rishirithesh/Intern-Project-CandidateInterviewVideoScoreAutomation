from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from config import Config


@dataclass
class VideoAnalysisResult:
    face_presence: float
    eye_contact: Optional[float]
    lighting: Optional[float]
    background_cleanliness: Optional[float]
    camera_stability: Optional[float]
    grooming: Optional[float]
    dressing: Optional[float]
    professionalism: Optional[float]
    face_confidence: float
    frames_sampled: int
    frames_analyzed: int
    face_detections: int
    dominant_clothing_color: Optional[str]


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _skin_mask_ratio(frame: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 10, 60], dtype=np.uint8)
    upper = np.array([25, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return float(np.count_nonzero(mask) / max(mask.size, 1))


def _extract_face_region(frame: np.ndarray, landmarks) -> Optional[Tuple[Tuple[int, int, int, int], np.ndarray]]:
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


def _score_lighting(brightness: float) -> float:
    # Brightness around the middle of the observed range is best; very dark or blown-out frames score lower.
    return _clamp(1.0 - abs(brightness - 0.58) / 0.45)


def _score_sharpness(face_roi: np.ndarray) -> float:
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return _clamp(variance / 180.0)


def _dominant_color_name(region: np.ndarray) -> Optional[str]:
    if region is None or region.size == 0:
        return None
    pixels = region.reshape(-1, 3)
    pixels = pixels[np.any(pixels > 20, axis=1)]
    if pixels.size == 0:
        return None
    b, g, r = np.mean(pixels, axis=0)
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
    hue, sat, val = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if val < 55:
        return "black"
    if sat < 35 and val > 190:
        return "white"
    if sat < 45:
        return "gray"
    if hue < 10 or hue >= 170:
        return "red"
    if hue < 25:
        return "orange"
    if hue < 35:
        return "yellow"
    if hue < 85:
        return "green"
    if hue < 125:
        return "blue"
    if hue < 155:
        return "purple"
    return "pink"


def _mean_optional(values: list[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


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
    frames_analyzed = 0
    lighting_values = []
    background_values = []
    centers = []
    eye_contact_values = []
    clothing_scores = []
    grooming_values = []
    clothing_colors = []
    frame_width = 0

    for timestamp in timestamps:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ret, frame = capture.read()
        if not ret or frame is None:
            continue

        frames_analyzed += 1
        frame_height, frame_width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_brightness = float(np.mean(gray) / 255.0)
        lighting_values.append(_score_lighting(frame_brightness))
        edges = cv2.Canny(gray, 80, 180)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            background_values.append(_clamp(1.0 - (np.count_nonzero(edges) / max(edges.size, 1)) * 1.7))
            continue

        face_detections += 1
        landmarks = results.multi_face_landmarks[0].landmark
        face_region_data = _extract_face_region(frame, landmarks)
        if face_region_data is None:
            background_values.append(_clamp(1.0 - (np.count_nonzero(edges) / max(edges.size, 1)) * 1.7))
            continue

        box, face_roi = face_region_data
        x, y, w_box, h_box = box
        centers.append((x + w_box / 2.0, y + h_box / 2.0))

        left_eye = np.mean([[landmarks[i].x, landmarks[i].y] for i in (33, 133)], axis=0)
        right_eye = np.mean([[landmarks[i].x, landmarks[i].y] for i in (362, 263)], axis=0)
        horizontal_offset = abs((left_eye[0] + right_eye[0]) / 2.0 - 0.5)
        eye_contact_values.append(_clamp(1.0 - horizontal_offset * 3.5))

        face_area_ratio = (w_box * h_box) / max(frame_height * frame_width, 1)
        face_visibility = _clamp(face_area_ratio * 10.0)
        face_brightness = float(np.mean(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)) / 255.0)
        grooming_values.append(_clamp((0.55 * _score_lighting(face_brightness)) + (0.45 * _score_sharpness(face_roi))) * face_visibility)

        shoulder_bottom = min(frame_height, int(y + h_box * 2.25))
        x_start = max(0, int(x - w_box * 0.15))
        x_end = min(frame_width, int(x + w_box * 1.15))
        clothing_region = frame[int(y + h_box * 0.95):shoulder_bottom, x_start:x_end]
        if clothing_region is not None and clothing_region.size > 0:
            skin_ratio = _skin_mask_ratio(clothing_region)
            clothing_scores.append(_clamp(1.0 - skin_ratio * 1.5))
            color_name = _dominant_color_name(clothing_region)
            if color_name:
                clothing_colors.append(color_name)

        mask = np.ones(edges.shape, dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x + w_box, y + h_box), 0, thickness=-1)
        background_values.append(_clamp(1.0 - (np.count_nonzero(edges[mask == 1]) / max(np.count_nonzero(mask == 1), 1)) * 1.7))

    capture.release()
    face_mesh.close()

    frames_sampled = len(timestamps)
    face_presence = _clamp(face_detections / max(frames_analyzed, 1)) if frames_analyzed else 0.0
    face_confidence = face_presence

    camera_stability = None
    if len(centers) > 1 and frame_width > 0:
        motions = [np.linalg.norm(np.array(centers[i]) - np.array(centers[i - 1])) for i in range(1, len(centers))]
        average_motion = float(np.mean(motions))
        camera_stability = _clamp(1.0 - (average_motion / frame_width) * 6.0)

    lighting = _mean_optional(lighting_values)
    background_cleanliness = _mean_optional(background_values)
    eye_contact = _mean_optional(eye_contact_values)
    grooming = _mean_optional(grooming_values)
    dressing = _mean_optional(clothing_scores)

    professional_components = [
        value for value in (eye_contact, lighting, background_cleanliness, camera_stability, grooming, dressing)
        if isinstance(value, (int, float))
    ]
    professionalism = _mean_optional(professional_components)
    dominant_clothing_color = max(set(clothing_colors), key=clothing_colors.count) if clothing_colors else None

    return VideoAnalysisResult(
        face_presence=round(face_presence, 3),
        eye_contact=round(eye_contact, 3) if eye_contact is not None else None,
        lighting=round(lighting, 3) if lighting is not None else None,
        background_cleanliness=round(background_cleanliness, 3) if background_cleanliness is not None else None,
        camera_stability=round(camera_stability, 3) if camera_stability is not None else None,
        grooming=round(grooming, 3) if grooming is not None else None,
        dressing=round(dressing, 3) if dressing is not None else None,
        professionalism=round(professionalism, 3) if professionalism is not None else None,
        face_confidence=round(face_confidence, 3),
        frames_sampled=frames_sampled,
        frames_analyzed=frames_analyzed,
        face_detections=face_detections,
        dominant_clothing_color=dominant_clothing_color,
    )
