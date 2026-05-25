import logging
import re
from typing import Dict, List

from modules.audio_analyzer import analyze_audio
from modules.evaluation_engine import build_category_scores
from modules.llm_client import InvalidLLMOutputError, ModelUnavailableError, generate_evaluation
from modules.video_analyzer import VideoAnalysisResult, analyze_video

logger = logging.getLogger(__name__)

FILLER_PHRASES = {
    "um", "uh", "like", "you know", "i mean", "sort of", "kind of",
    "basically", "actually", "right", "well", "so"
}

VAGUE_TERMS = {
    "stuff", "things", "something", "somehow", "etc", "many things",
    "good", "nice", "simple", "easy", "basically", "various"
}

TECHNICAL_TERMS = {
    "api", "database", "schema", "query", "index", "cache", "authentication",
    "authorization", "jwt", "oauth", "backend", "frontend", "deployment",
    "docker", "kubernetes", "cloud", "pipeline", "model", "algorithm",
    "latency", "scalability", "microservice", "server", "client", "endpoint",
    "testing", "unit test", "integration", "architecture", "framework",
    "react", "flask", "django", "node", "python", "java", "sql", "nosql"
}

IMPLEMENTATION_SIGNALS = {
    "implemented", "built", "designed", "created", "integrated", "configured",
    "optimized", "debugged", "handled", "stored", "validated", "deployed",
    "tested", "refactored", "connected", "wrote", "used"
}

DEBUGGING_SIGNALS = {
    "debug", "bug", "issue", "error", "failure", "fixed", "troubleshoot",
    "logs", "exception", "root cause"
}

TRADEOFF_SIGNALS = {
    "tradeoff", "because", "instead", "compared", "latency", "cost",
    "scalable", "scale", "performance", "security", "reliability",
    "maintainability", "constraint", "decision"
}

OWNERSHIP_SIGNALS = {
    "i built", "i implemented", "i designed", "i created", "i integrated",
    "my role", "i was responsible", "i handled", "i worked on", "i deployed",
    "i tested", "i fixed"
}

PRODUCTION_SIGNALS = {
    "deploy", "production", "monitoring", "logging", "testing", "ci/cd",
    "security", "scalability", "performance", "availability", "rollback",
    "environment", "database migration"
}

KPI_SCHEMA = {
    "communication_skills": "Communication Skills",
    "technical_skills": "Technical Skills",
    "project_understanding": "Project Understanding",
}


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [sentence for sentence in sentences if sentence]


def _split_words(text: str) -> List[str]:
    return re.findall(r"\b[\w']+\b", text.lower())


def _count_phrase_matches(text: str, phrases: set) -> int:
    lowered = f" {text.lower()} "
    total = 0
    for phrase in phrases:
        if " " in phrase:
            total += lowered.count(f" {phrase} ")
        else:
            total += len(re.findall(rf"\b{re.escape(phrase)}\b", lowered))
    return total


def _get_repeated_phrases(words: List[str]) -> int:
    phrases = {}
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i + 1]}"
        phrases[phrase] = phrases.get(phrase, 0) + 1
    return sum(count - 1 for count in phrases.values() if count > 1)


def _has_video_evidence(video_metrics: VideoAnalysisResult) -> bool:
    return video_metrics.face_presence >= 0.45 and video_metrics.face_confidence >= 0.45


def _build_prompt(transcript: str, audio_metrics: Dict[str, float], video_metrics: VideoAnalysisResult) -> str:
    video_instruction = (
        "Video evidence is available. You may mention only observable presentation issues if they are directly supported by the metrics."
        if _has_video_evidence(video_metrics)
        else "Video evidence is not reliable enough for grooming or professional presence scoring. Do not score or discuss grooming."
    )

    return f"""
You are a strict technical interviewer and recruiter-system evaluator.
Evaluate only the candidate evidence in the transcript and the provided audio/video metrics.
Do not reward buzzword lists unless the candidate explains implementation, ownership, tradeoffs, debugging, APIs, databases, deployment, or architecture.
Do not infer skills, experience, grooming, confidence, project ownership, or job fit without evidence.
Be realistic: 5-6 is average/borderline, 7-8 is genuinely good, 9-10 is exceptional and requires strong transcript evidence.

Return ONLY valid JSON. No markdown. No extra keys outside this schema:
{{
  "kpis": {{
    "communication_skills": {{
      "score": number from 0 to 10,
      "evidence": ["short transcript quote or observation"],
      "strengths": ["specific strength"],
      "weaknesses": ["specific weakness"],
      "reasoning": "detailed recruiter reasoning"
    }},
    "technical_skills": {{
      "score": number from 0 to 10,
      "evidence": ["short transcript quote or observation"],
      "strengths": ["specific strength"],
      "weaknesses": ["specific weakness"],
      "reasoning": "detailed recruiter reasoning"
    }},
    "project_understanding": {{
      "score": number from 0 to 10,
      "evidence": ["short transcript quote or observation"],
      "strengths": ["specific strength"],
      "weaknesses": ["specific weakness"],
      "reasoning": "detailed recruiter reasoning"
    }}
  }},
  "recruiter_summary": {{
    "summary": "final recruiter summary tied to the KPI evidence",
    "biggest_strengths": ["specific strength"],
    "biggest_risks": ["specific risk"],
    "confidence": "low, medium, or high"
  }}
}}

Scoring rubric:
9-10: exceptional production-grade thinking, clear ownership, deep implementation detail.
7-8: hireable, practical understanding, solid communication, minor gaps.
5-6: average/borderline, surface-level or incomplete depth.
3-4: weak, vague, shallow reasoning, poor articulation.
0-2: very poor, no meaningful evidence or likely copied/tutorial-level understanding.

Communication assessment must consider clarity, fluency, filler usage, sentence structure, concise speaking, professionalism, and technical articulation.
Technical assessment must consider implementation depth, APIs/databases, debugging, architecture, deployment/devops, scalability, framework knowledge, and practical problem solving.
Project understanding must assess whether the candidate appears to have truly built or owned the work, including constraints, tradeoffs, feature reasoning, production readiness, and business awareness.
{video_instruction}

Audio metrics:
- duration_seconds: {audio_metrics.get("duration")}
- energy_score_0_to_1: {audio_metrics.get("energy_score")}
- clarity_score_0_to_1: {audio_metrics.get("clarity_score")}
- silence_ratio: {audio_metrics.get("silence_ratio")}

Video metrics:
- face_presence_0_to_1: {video_metrics.face_presence}
- face_confidence_0_to_1: {video_metrics.face_confidence}
- eye_contact_0_to_1: {video_metrics.eye_contact}
- lighting_0_to_1: {video_metrics.lighting}
- background_cleanliness_0_to_1: {video_metrics.background_cleanliness}
- camera_stability_0_to_1: {video_metrics.camera_stability}

Transcript:
{transcript[:3500]}
""".strip()


def _build_transcript_metrics(transcript: str, duration: float) -> Dict[str, float]:
    words = _split_words(transcript)
    sentences = _split_sentences(transcript)
    word_count = len(words)
    unique_ratio = len(set(words)) / max(word_count, 1)
    filler_count = _count_phrase_matches(transcript, FILLER_PHRASES)
    technical_term_count = _count_phrase_matches(transcript, TECHNICAL_TERMS)
    implementation_signal_count = _count_phrase_matches(transcript, IMPLEMENTATION_SIGNALS)
    debugging_signal_count = _count_phrase_matches(transcript, DEBUGGING_SIGNALS)
    tradeoff_signal_count = _count_phrase_matches(transcript, TRADEOFF_SIGNALS)
    ownership_signal_count = _count_phrase_matches(transcript, OWNERSHIP_SIGNALS)
    production_signal_count = _count_phrase_matches(transcript, PRODUCTION_SIGNALS)
    vague_term_count = _count_phrase_matches(transcript, VAGUE_TERMS)
    repeated_phrases = _get_repeated_phrases(words)
    average_sentence_length = float(word_count) / max(len(sentences), 1)
    words_per_minute = float(word_count) / max(duration, 1e-3) * 60.0
    evidence_signal_count = implementation_signal_count + tradeoff_signal_count + debugging_signal_count
    buzzword_ratio = max(0, technical_term_count - evidence_signal_count) / max(technical_term_count, 1)

    return {
        "word_count": word_count,
        "sentence_count": len(sentences),
        "unique_ratio": round(unique_ratio, 3),
        "filler_count": filler_count,
        "filler_rate": round(filler_count / max(word_count, 1), 3),
        "vague_term_count": vague_term_count,
        "technical_term_count": technical_term_count,
        "implementation_signal_count": implementation_signal_count,
        "debugging_signal_count": debugging_signal_count,
        "tradeoff_signal_count": tradeoff_signal_count,
        "ownership_signal_count": ownership_signal_count,
        "production_signal_count": production_signal_count,
        "buzzword_ratio": round(buzzword_ratio, 3),
        "repeated_phrases": repeated_phrases,
        "average_sentence_length": round(average_sentence_length, 1),
        "words_per_minute": round(words_per_minute, 1),
        "duration": round(duration, 1),
    }


def _require_list(value: object, field_name: str) -> List[str]:
    if not isinstance(value, list):
        raise InvalidLLMOutputError(f"LLM KPI field '{field_name}' must be a list.")
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_llm_kpis(data: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    if not isinstance(data, dict) or not isinstance(data.get("kpis"), dict):
        raise InvalidLLMOutputError("LLM output must include a 'kpis' object.")

    normalized = {}
    for raw_key, display_name in KPI_SCHEMA.items():
        raw_kpi = data["kpis"].get(raw_key)
        if not isinstance(raw_kpi, dict):
            raise InvalidLLMOutputError(f"LLM output missing KPI '{raw_key}'.")

        score = raw_kpi.get("score")
        if not isinstance(score, (int, float)):
            raise InvalidLLMOutputError(f"LLM KPI '{raw_key}' must include a numeric score.")

        evidence = _require_list(raw_kpi.get("evidence"), f"{raw_key}.evidence")
        reasoning = raw_kpi.get("reasoning")
        if not evidence:
            raise InvalidLLMOutputError(f"LLM KPI '{raw_key}' must include transcript-grounded evidence.")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise InvalidLLMOutputError(f"LLM KPI '{raw_key}' must include detailed reasoning.")

        normalized[display_name] = {
            "score": max(0.0, min(10.0, float(score))),
            "evidence": evidence[:5],
            "strengths": _require_list(raw_kpi.get("strengths"), f"{raw_key}.strengths")[:5],
            "weaknesses": _require_list(raw_kpi.get("weaknesses"), f"{raw_key}.weaknesses")[:5],
            "reasoning": reasoning.strip(),
            "available": True,
        }

    return normalized


def _build_recruiter_summary(llm_response: Dict[str, object], decision: str, final_score: float, strengths: List[str], weaknesses: List[str], confidence: float) -> str:
    summary_data = llm_response.get("recruiter_summary")
    model_summary = ""
    risks = []
    positives = []
    if isinstance(summary_data, dict):
        if isinstance(summary_data.get("summary"), str):
            model_summary = summary_data["summary"].strip()
        if isinstance(summary_data.get("biggest_risks"), list):
            risks = [str(item).strip() for item in summary_data["biggest_risks"] if str(item).strip()]
        if isinstance(summary_data.get("biggest_strengths"), list):
            positives = [str(item).strip() for item in summary_data["biggest_strengths"] if str(item).strip()]

    parts = []
    if model_summary:
        parts.append(model_summary)
    parts.append(
        "Decision: %s at %.1f/10 based on weighted evidence across communication, technical skill, and project ownership."
        % (decision, final_score)
    )
    if positives:
        parts.append("Biggest strengths: %s." % "; ".join(positives[:3]))
    elif strengths:
        parts.append("Biggest strengths: %s." % ", ".join(strengths[:3]))
    if risks:
        parts.append("Biggest risks: %s." % "; ".join(risks[:3]))
    elif weaknesses:
        parts.append("Biggest risks: %s." % ", ".join(weaknesses[:3]))
    parts.append("Evaluation confidence: %.1f%%." % confidence)
    return " ".join(parts)


def evaluate_candidate(transcript: str, audio_path: str, video_path: str) -> dict:
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("Transcript is empty; cannot evaluate candidate.")

    audio_metrics = analyze_audio(audio_path)
    video_metrics = analyze_video(video_path)
    transcript_metrics = _build_transcript_metrics(transcript, audio_metrics["duration"])

    try:
        llm_response = generate_evaluation(_build_prompt(transcript, audio_metrics, video_metrics))
    except ModelUnavailableError as exc:
        raise RuntimeError("LLM evaluation failed: %s" % str(exc)) from exc

    llm_kpis = _normalize_llm_kpis(llm_response)
    category_scores, final_score, decision, strengths, weaknesses, confidence = build_category_scores(
        llm_kpis=llm_kpis,
        audio_metrics=audio_metrics,
        video_metrics=video_metrics,
        transcript_metrics=transcript_metrics,
    )

    kpi_reasons = {
        name: [data.get("reasoning", "")]
        for name, data in category_scores.items()
        if data.get("reasoning")
    }
    ai_feedback = _build_recruiter_summary(llm_response, decision, final_score, strengths, weaknesses, confidence)

    return {
        "final_score": final_score,
        "decision": decision,
        "detailed_scores": category_scores,
        "kpi_reasons": kpi_reasons,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "confidence": confidence,
        "ai_feedback": ai_feedback,
        "audio_metrics": audio_metrics,
        "video_metrics": {
            "face_presence": video_metrics.face_presence,
            "eye_contact": video_metrics.eye_contact,
            "lighting": video_metrics.lighting,
            "background_cleanliness": video_metrics.background_cleanliness,
            "camera_stability": video_metrics.camera_stability,
            "grooming": video_metrics.grooming if _has_video_evidence(video_metrics) else None,
            "dressing": video_metrics.dressing if _has_video_evidence(video_metrics) else None,
            "professionalism": video_metrics.professionalism if _has_video_evidence(video_metrics) else None,
            "face_confidence": video_metrics.face_confidence,
            "professional_presence_available": _has_video_evidence(video_metrics),
        },
        "transcript_metrics": transcript_metrics,
        "raw_llm_response": llm_response.get("raw_llm_response", None),
    }
