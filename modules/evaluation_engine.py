import math
from typing import Dict, List, Tuple

from modules.video_analyzer import VideoAnalysisResult

CATEGORY_WEIGHTS = {
    "Communication": 18,
    "Confidence": 15,
    "Content": 18,
    "Structure": 14,
    "Professional Presence": 15,
    "Timing": 10,
    "Originality": 10,
}


def clamp(value: float, minimum: float = 1.0, maximum: float = 10.0) -> float:
    return max(minimum, min(maximum, value))


def normalize_to_score(value: float, ideal_low: float, ideal_high: float, minimum: float, maximum: float) -> float:
    if value <= minimum or value >= maximum:
        return 1.0
    if ideal_low <= value <= ideal_high:
        return 10.0
    if value < ideal_low:
        return 1.0 + 9.0 * ((value - minimum) / (ideal_low - minimum))
    return 1.0 + 9.0 * ((maximum - value) / (maximum - ideal_high))


def repeated_phrase_count(words: List[str]) -> int:
    phrases = {}
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i + 1]}"
        phrases[phrase] = phrases.get(phrase, 0) + 1
    return sum(count - 1 for count in phrases.values() if count > 1)


def safe_average(values: List[float]) -> float:
    filtered = [v for v in values if isinstance(v, (int, float))]
    if not filtered:
        return 1.0
    return sum(filtered) / len(filtered)


def build_category_scores(
    llm_scores: Dict[str, float],
    audio_metrics: Dict[str, float],
    video_metrics: VideoAnalysisResult,
    transcript_metrics: Dict[str, float],
    transcript_text: str,
) -> Tuple[Dict[str, Dict[str, object]], float, str, List[str], List[str], float]:
    words = transcript_metrics["word_count"]
    sentences = transcript_metrics["sentence_count"]
    unique_ratio = transcript_metrics["unique_ratio"]
    repeated_phrases = transcript_metrics["repeated_phrases"]
    duration = transcript_metrics["duration"]
    wpm = transcript_metrics["words_per_minute"]

    communication_score = clamp(
        safe_average([
            llm_scores.get("communication", 5.0),
            normalize_to_score(wpm, 110, 150, 60, 200) * 1.0,
            audio_metrics.get("clarity_score", 0.8) * 10,
        ])
    )

    confidence_score = clamp(
        safe_average([
            llm_scores.get("confidence", 5.0),
            audio_metrics.get("energy_score", 0.8) * 10,
            video_metrics.professionalism * 10,
        ])
    )

    density_score = clamp(
        normalize_to_score(words, 80, 160, 20, 260)
    )
    vocabulary_score = clamp(1.0 + 9.0 * max(0.0, min(unique_ratio, 1.0) - 0.4) / 0.6)
    content_score = clamp(
        safe_average([
            llm_scores.get("content", 5.0),
            density_score,
            vocabulary_score,
            video_metrics.grooming * 10,
        ])
    )

    structure_score = clamp(
        safe_average([
            llm_scores.get("structure", 5.0),
            normalize_to_score(sentences, 3, 8, 1, 15),
            normalize_to_score(transcript_metrics["average_sentence_length"], 12, 25, 5, 40),
            video_metrics.lighting * 10,
        ])
    )

    professionalism_score = clamp(
        safe_average([
            video_metrics.professionalism * 10,
            video_metrics.eye_contact * 10,
            video_metrics.background_cleanliness * 10,
            video_metrics.camera_stability * 10,
        ])
    )

    timing_score = clamp(
        safe_average([
            normalize_to_score(duration, 40, 120, 15, 240),
            normalize_to_score(wpm, 110, 150, 60, 200),
        ])
    )

    originality_score = clamp(10.0 - min(4.0, repeated_phrases * 0.8))

    category_values = {
        "Communication": communication_score,
        "Confidence": confidence_score,
        "Content": content_score,
        "Structure": structure_score,
        "Professional Presence": professionalism_score,
        "Timing": timing_score,
        "Originality": originality_score,
    }

    detailed_scores = {}
    reasons = {}
    strengths = []
    weaknesses = []

    for category, value in category_values.items():
        weight = CATEGORY_WEIGHTS[category]
        detailed_scores[category] = {
            "score": round(value, 1),
            "weight": weight,
        }

        if value >= 8.0:
            strengths.append(category)
        if value <= 5.0:
            weaknesses.append(category)

        if category == "Communication":
            reason = "Clear pace and strong delivery." if value >= 8 else (
                "Speech can improve with more structure and fewer fillers." if value >= 5 else "Communication is too slow or inconsistent."
            )
        elif category == "Confidence":
            reason = "Confident delivery and strong visual presence." if value >= 8 else (
                "Moderate confidence; audio energy and posture could improve." if value >= 5 else "Candidate appears underconfident or uncertain."
            )
        elif category == "Content":
            reason = "Strong answer depth with varied vocabulary." if value >= 8 else (
                "Content is serviceable but could be more concise and focused." if value >= 5 else "Answer lacks depth and clarity."
            )
        elif category == "Structure":
            reason = "Well-organized response with clear transitions." if value >= 8 else (
                "Structure is acceptable but some points feel disconnected." if value >= 5 else "Response needs clearer structure."
            )
        elif category == "Professional Presence":
            reason = "Solid screen presence, good lighting, and steady camera." if value >= 8 else (
                "Visual presence is adequate but could improve in lighting and stability." if value >= 5 else "Video quality and on-screen presence are weak."
            )
        elif category == "Timing":
            reason = "Response length is well matched to the question." if value >= 8 else (
                "Minor timing issues; could be more concise or expansive." if value >= 5 else "Answer is too short or too long for the task."
            )
        else:
            reason = "Original and varied language use." if value >= 8 else (
                "Some repetition appears in the response." if value >= 5 else "The response is overly repetitive."
            )

        reasons[category] = [reason]

    total_weighted = sum((score / 10.0) * CATEGORY_WEIGHTS[name] for name, score in category_values.items())
    final_score = round(total_weighted / 10.0, 1)
    confidence_rating = round(min(1.0, max(0.0, final_score / 10.0)) * 100, 1)

    if final_score >= 7.5:
        decision = "SELECT"
    elif final_score <= 6.5:
        decision = "REJECT"
    else:
        decision = "BORDERLINE"

    if not strengths:
        strengths = [max(category_values, key=category_values.get)]
    if not weaknesses:
        weaknesses = [min(category_values, key=category_values.get)]

    return (
        detailed_scores,
        final_score,
        decision,
        reasons,
        strengths,
        weaknesses,
        confidence_rating,
    )
