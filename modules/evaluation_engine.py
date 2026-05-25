from typing import Dict, List, Optional, Tuple

from modules.video_analyzer import VideoAnalysisResult

KPI_WEIGHTS = {
    "Communication Skills": 25,
    "Technical Skills": 30,
    "Project Understanding": 25,
    "Professional Presence": 10,
    "Interview Readiness": 10,
}

HIRING_THRESHOLDS = {
    "hire": 6.0,
    "borderline": 5.0,
}


def clamp(value: float, minimum: float = 0.0, maximum: float = 10.0) -> float:
    return max(minimum, min(maximum, value))


def normalize_to_score(value: float, ideal_low: float, ideal_high: float, minimum: float, maximum: float) -> float:
    if value <= minimum or value >= maximum:
        return 0.0
    if ideal_low <= value <= ideal_high:
        return 10.0
    if value < ideal_low:
        return 10.0 * ((value - minimum) / (ideal_low - minimum))
    return 10.0 * ((maximum - value) / (maximum - ideal_high))


def safe_average(values: List[float]) -> float:
    filtered = [float(v) for v in values if isinstance(v, (int, float))]
    if not filtered:
        return 0.0
    return sum(filtered) / len(filtered)


def _score_with_penalty(base_score: float, penalty: float) -> float:
    return clamp(base_score - max(0.0, penalty))


def _has_video_evidence(video_metrics: VideoAnalysisResult) -> bool:
    return (
        video_metrics.face_confidence >= 0.45
        and video_metrics.face_presence >= 0.45
    )


def _communication_penalty(transcript_metrics: Dict[str, float], audio_metrics: Dict[str, float]) -> Tuple[float, List[str]]:
    reasons = []
    penalty = 0.0

    word_count = transcript_metrics["word_count"]
    filler_rate = transcript_metrics["filler_rate"]
    repeated_phrases = transcript_metrics["repeated_phrases"]
    average_sentence_length = transcript_metrics["average_sentence_length"]
    wpm = transcript_metrics["words_per_minute"]
    clarity_score = audio_metrics.get("clarity_score")

    if word_count < 45:
        penalty += 1.6
        reasons.append("Transcript is very short, limiting confidence in communication assessment.")
    if filler_rate >= 0.08:
        penalty += 1.4
        reasons.append("Filler word rate is high for an interview answer.")
    elif filler_rate >= 0.04:
        penalty += 0.7
        reasons.append("Filler words are noticeable and slightly reduce fluency.")
    if repeated_phrases >= 6:
        penalty += 0.8
        reasons.append("Repeated phrases suggest looping or rambling.")
    if average_sentence_length > 32:
        penalty += 0.7
        reasons.append("Long average sentence length suggests less concise delivery.")
    if wpm < 80 or wpm > 185:
        penalty += 0.7
        reasons.append("Speaking pace appears outside a recruiter-friendly range.")
    if isinstance(clarity_score, (int, float)) and clarity_score < 0.35:
        penalty += 1.0
        reasons.append("Audio clarity is low, reducing confidence in spoken delivery.")

    return penalty, reasons


def _technical_penalty(transcript_metrics: Dict[str, float]) -> Tuple[float, List[str]]:
    reasons = []
    penalty = 0.0

    if transcript_metrics["technical_term_count"] < 3:
        penalty += 1.4
        reasons.append("Transcript contains limited concrete technical vocabulary.")
    if transcript_metrics["implementation_signal_count"] < 2:
        penalty += 1.6
        reasons.append("Few implementation-level details were detected.")
    if transcript_metrics["debugging_signal_count"] == 0:
        penalty += 0.6
        reasons.append("No debugging, failure, or troubleshooting evidence was detected.")
    if transcript_metrics["tradeoff_signal_count"] == 0:
        penalty += 0.6
        reasons.append("No tradeoff or architecture decision evidence was detected.")
    if transcript_metrics["technical_term_count"] >= 4 and transcript_metrics["buzzword_ratio"] > 0.6:
        penalty += 1.2
        reasons.append("Technical language appears buzzword-heavy compared with implementation detail.")

    return penalty, reasons


def _project_penalty(transcript_metrics: Dict[str, float]) -> Tuple[float, List[str]]:
    reasons = []
    penalty = 0.0

    if transcript_metrics["ownership_signal_count"] < 2:
        penalty += 1.5
        reasons.append("Ownership signals are weak or infrequent.")
    if transcript_metrics["implementation_signal_count"] < 2:
        penalty += 1.0
        reasons.append("Project explanation lacks concrete implementation evidence.")
    if transcript_metrics["production_signal_count"] == 0:
        penalty += 0.7
        reasons.append("No production, deployment, testing, or scalability evidence was detected.")
    if transcript_metrics["vague_term_count"] >= 8:
        penalty += 0.8
        reasons.append("Frequent vague terms reduce confidence in genuine project ownership.")

    return penalty, reasons


def _professional_presence_score(video_metrics: VideoAnalysisResult) -> Optional[float]:
    if not _has_video_evidence(video_metrics):
        return None
    return clamp(safe_average([
        video_metrics.eye_contact * 10,
        video_metrics.lighting * 10,
        video_metrics.background_cleanliness * 10,
        video_metrics.camera_stability * 10,
        video_metrics.dressing * 10,
    ]))


def _presence_reasons(video_metrics: VideoAnalysisResult, available: bool) -> Tuple[List[str], List[str], List[str]]:
    if not available:
        return (
            ["Video face evidence was insufficient for a reliable professional presence assessment."],
            [],
            ["Professional presence was marked unavailable instead of inferred."]
        )

    evidence = [
        "Video metrics: eye contact %.1f/10, lighting %.1f/10, background %.1f/10, camera stability %.1f/10."
        % (
            video_metrics.eye_contact * 10,
            video_metrics.lighting * 10,
            video_metrics.background_cleanliness * 10,
            video_metrics.camera_stability * 10,
        )
    ]
    strengths = []
    weaknesses = []
    if video_metrics.eye_contact >= 0.65:
        strengths.append("Maintains reasonably direct camera alignment.")
    elif video_metrics.eye_contact < 0.45:
        weaknesses.append("Eye contact/camera alignment appears weak.")
    if video_metrics.lighting >= 0.65:
        strengths.append("Lighting supports interview visibility.")
    elif video_metrics.lighting < 0.45:
        weaknesses.append("Lighting may reduce professional presentation quality.")
    if video_metrics.background_cleanliness < 0.45:
        weaknesses.append("Background appears visually busy.")
    if video_metrics.camera_stability < 0.45:
        weaknesses.append("Camera stability is weak.")
    return evidence, strengths, weaknesses


def _append_penalty_evidence(kpi: Dict[str, object], penalty_reasons: List[str]) -> None:
    if not penalty_reasons:
        return
    kpi.setdefault("weaknesses", [])
    kpi.setdefault("evidence", [])
    for reason in penalty_reasons:
        if reason not in kpi["weaknesses"]:
            kpi["weaknesses"].append(reason)
        if reason not in kpi["evidence"]:
            kpi["evidence"].append(reason)


def _cap_for_weak_evidence(score: float, kpi: Dict[str, object], transcript_metrics: Dict[str, float]) -> float:
    evidence_items = [item for item in kpi.get("evidence", []) if isinstance(item, str) and item.strip()]
    if len(evidence_items) < 1:
        return min(score, 4.0)
    if transcript_metrics["word_count"] < 35:
        return min(score, 5.0)
    return score


def build_category_scores(
    llm_kpis: Dict[str, Dict[str, object]],
    audio_metrics: Dict[str, float],
    video_metrics: VideoAnalysisResult,
    transcript_metrics: Dict[str, float],
) -> Tuple[Dict[str, Dict[str, object]], float, str, List[str], List[str], float]:
    detailed_scores: Dict[str, Dict[str, object]] = {}

    communication = dict(llm_kpis["Communication Skills"])
    communication_penalty, communication_penalty_reasons = _communication_penalty(transcript_metrics, audio_metrics)
    communication["score"] = _cap_for_weak_evidence(
        _score_with_penalty(float(communication["score"]), communication_penalty),
        communication,
        transcript_metrics,
    )
    _append_penalty_evidence(communication, communication_penalty_reasons)

    technical = dict(llm_kpis["Technical Skills"])
    technical_penalty, technical_penalty_reasons = _technical_penalty(transcript_metrics)
    technical["score"] = _cap_for_weak_evidence(
        _score_with_penalty(float(technical["score"]), technical_penalty),
        technical,
        transcript_metrics,
    )
    _append_penalty_evidence(technical, technical_penalty_reasons)

    project = dict(llm_kpis["Project Understanding"])
    project_penalty, project_penalty_reasons = _project_penalty(transcript_metrics)
    project["score"] = _cap_for_weak_evidence(
        _score_with_penalty(float(project["score"]), project_penalty),
        project,
        transcript_metrics,
    )
    _append_penalty_evidence(project, project_penalty_reasons)

    presence_score = _professional_presence_score(video_metrics)
    presence_available = presence_score is not None
    presence_evidence, presence_strengths, presence_weaknesses = _presence_reasons(video_metrics, presence_available)
    presence = {
        "score": presence_score,
        "available": presence_available,
        "weight": KPI_WEIGHTS["Professional Presence"],
        "evidence": presence_evidence,
        "strengths": presence_strengths,
        "weaknesses": presence_weaknesses,
        "reasoning": (
            "Professional presence was evaluated from measurable video quality and face/camera signals."
            if presence_available
            else "Professional presence is not scored because reliable video evidence was unavailable."
        ),
    }

    readiness_base = safe_average([
        communication["score"],
        technical["score"],
        project["score"],
    ])
    readiness_penalty = 0.0
    readiness_evidence = [
        "Readiness combines communication, technical depth, and project ownership rather than adding a separate static score."
    ]
    if transcript_metrics["word_count"] < 80:
        readiness_penalty += 0.8
        readiness_evidence.append("Short transcript reduces hiring confidence.")
    if technical["score"] < 5.0 or project["score"] < 5.0:
        readiness_penalty += 0.8
        readiness_evidence.append("Core technical/project weakness creates interview risk.")
    readiness = {
        "score": _score_with_penalty(readiness_base, readiness_penalty),
        "available": True,
        "weight": KPI_WEIGHTS["Interview Readiness"],
        "evidence": readiness_evidence,
        "strengths": [],
        "weaknesses": [],
        "reasoning": "This score reflects whether the candidate looks ready for recruiter advancement after applying the KPI evidence.",
    }

    kpis = {
        "Communication Skills": communication,
        "Technical Skills": technical,
        "Project Understanding": project,
        "Professional Presence": presence,
        "Interview Readiness": readiness,
    }

    active_weight = 0
    weighted_total = 0.0
    for name, data in kpis.items():
        weight = KPI_WEIGHTS[name]
        data["weight"] = weight
        data["available"] = bool(data.get("available", True))
        score = data.get("score")
        if data["available"] and isinstance(score, (int, float)):
            active_weight += weight
            weighted_total += (float(score) / 10.0) * weight
            data["score"] = round(clamp(float(score)), 1)
        else:
            data["score"] = None
        detailed_scores[name] = data

    final_score = round((weighted_total / active_weight) * 10.0, 1) if active_weight else 0.0

    if final_score >= HIRING_THRESHOLDS["hire"]:
        decision = "HIRE"
    elif final_score >= HIRING_THRESHOLDS["borderline"]:
        decision = "BORDERLINE"
    else:
        decision = "REJECT"

    strengths = [
        name for name, data in detailed_scores.items()
        if data.get("available") and isinstance(data.get("score"), (int, float)) and data["score"] >= 7.0
    ]
    weaknesses = [
        name for name, data in detailed_scores.items()
        if data.get("available") and isinstance(data.get("score"), (int, float)) and data["score"] < 5.0
    ]

    if not strengths:
        scored = {name: data["score"] for name, data in detailed_scores.items() if data.get("score") is not None}
        strengths = [max(scored, key=scored.get)] if scored else []
    if not weaknesses:
        scored = {name: data["score"] for name, data in detailed_scores.items() if data.get("score") is not None}
        weaknesses = [min(scored, key=scored.get)] if scored else []

    confidence_factors = [
        min(1.0, transcript_metrics["word_count"] / 160.0),
        1.0 - min(0.7, transcript_metrics["filler_rate"] * 3.0),
        min(1.0, max(0.0, audio_metrics.get("clarity_score", 0.0))),
    ]
    if presence_available:
        confidence_factors.append(video_metrics.face_confidence)
    confidence_rating = round(safe_average(confidence_factors) * 100.0, 1)

    return detailed_scores, final_score, decision, strengths, weaknesses, confidence_rating
