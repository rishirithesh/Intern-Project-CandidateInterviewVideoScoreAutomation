import logging
import re
from typing import Dict, List

from modules.audio_analyzer import analyze_audio
from modules.evaluation_engine import build_category_scores
from modules.llm_client import ModelUnavailableError, generate_evaluation
from modules.video_analyzer import VideoAnalysisResult, analyze_video

logger = logging.getLogger(__name__)

FILLER_WORDS = {'um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'right', 'well'}


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [sentence for sentence in sentences if sentence]


def _split_words(text: str) -> List[str]:
    return re.findall(r"\b[\w']+\b", text.lower())


def _get_repeated_phrases(words: List[str]) -> int:
    phrases = {}
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i + 1]}"
        phrases[phrase] = phrases.get(phrase, 0) + 1
    return sum(count - 1 for count in phrases.values() if count > 1)


def _build_prompt(transcript: str, audio_metrics: Dict[str, float], video_metrics: VideoAnalysisResult) -> str:
    energy_score = int(audio_metrics.get('energy', 5) * 10)
    clarity_score = int(audio_metrics.get('clarity_score', 5) * 10)
    prof_score = int(video_metrics.professionalism * 10)
    
    return (
        'You are evaluating a candidate video for professional readiness using only visible and audible evidence. '
        'Do not infer anything that cannot be clearly seen or heard. Return ONLY valid JSON with flat integer scores 1-10 and explicit comments for each KPI. '
        'Use double quotes for all keys and string values. Do not use markdown, explanations, or extra fields.\n'
        'Example exact format:\n'
        '{"communication": 7, "confidence": 8, "content": 6, "structure": 7, '
        '"communication_reason": "Clear and focused delivery.", '
        '"confidence_reason": "Steady tone and direct eye contact.", '
        '"content_reason": "Organized response with relevant examples.", '
        '"structure_reason": "Logical flow with good transitions.", '
        '"reason": "Overall solid interview performance."}\n\n'
        'Transcript excerpt (first 500 chars):\n'
        f'{transcript[:500]}\n\n'
        f'Audio energy: {energy_score}/10, audio clarity: {clarity_score}/10, visible professionalism: {prof_score}/10.\n'
        'Provide one sentence comments for each KPI field: communication_reason, confidence_reason, content_reason, and structure_reason. '
        'Return JSON only.'
    )


def _normalize_llm_scores(data: Dict[str, object]) -> Dict[str, float]:
    result = {}
    for key in ('communication', 'confidence', 'content', 'structure'):
        value = data.get(key)
        if isinstance(value, (int, float)):
            result[key] = float(max(1.0, min(10.0, value)))
        elif isinstance(value, dict):
            numeric = next((v for v in value.values() if isinstance(v, (int, float))), None)
            if numeric is not None:
                result[key] = float(max(1.0, min(10.0, numeric)))
                continue
            logger.warning('LLM score for %s is nested but contains no numeric value (%r); defaulting to 5.0', key, value)
            result[key] = 5.0
        else:
            logger.warning('LLM score for %s is invalid (%r); defaulting to 5.0', key, value)
            result[key] = 5.0
    return result


def _build_transcript_metrics(transcript: str, duration: float) -> Dict[str, float]:
    words = _split_words(transcript)
    sentences = _split_sentences(transcript)
    word_count = len(words)
    unique_ratio = len(set(words)) / max(word_count, 1)
    repeated_phrases = _get_repeated_phrases(words)
    average_sentence_length = float(word_count) / max(len(sentences), 1)
    words_per_minute = float(word_count) / max(duration, 1e-3) * 60.0

    return {
        'word_count': word_count,
        'sentence_count': len(sentences),
        'unique_ratio': round(unique_ratio, 3),
        'repeated_phrases': repeated_phrases,
        'average_sentence_length': round(average_sentence_length, 1),
        'words_per_minute': round(words_per_minute, 1),
        'duration': round(duration, 1),
    }


def _sanitize_ai_feedback(llm_response: Dict[str, object], llm_scores: Dict[str, float]) -> str:
    """Return a human-readable AI feedback string derived from the LLM response.

    - Prefer a non-empty string in `reason` or equivalent overall feedback fields.
    - If `reason` is a dict, join its string values.
    - As a last resort, synthesize feedback from the numeric scores.
    """
    if not isinstance(llm_response, dict):
        return 'No AI feedback available.'

    for field in ('reason', 'overall_reason', 'feedback', 'comment', 'comments'):
        content = llm_response.get(field)
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, dict):
            parts = [str(v).strip() for v in content.values() if isinstance(v, str) and v.strip()]
            if parts:
                return ' '.join(parts)

    # If no overall reason, build feedback from KPI-specific comments.
    kpi_comments = []
    for prefix, label in [
        ('content', 'Content'),
        ('confidence', 'Confidence'),
        ('communication', 'Communication'),
        ('structure', 'Structure'),
    ]:
        for key in (f'{prefix}_reason', f'{prefix}_comment', f'{prefix}_feedback'):
            value = llm_response.get(key)
            if isinstance(value, str) and value.strip():
                kpi_comments.append(f"{label}: {value.strip()}")
                break
    if kpi_comments:
        return ' '.join(kpi_comments)

    # Fallback: summarize the numeric scores into a short sentence
    if llm_scores:
        parts = []
        for k in ('communication', 'confidence', 'content', 'structure'):
            v = llm_scores.get(k)
            if v is not None:
                parts.append(f"{k}={float(v):.1f}")
        if parts:
            return 'LLM feedback (scores): ' + ', '.join(parts)

    return 'No AI feedback available.'


def evaluate_candidate(transcript: str, audio_path: str, video_path: str) -> dict:
    transcript = transcript.strip()
    if not transcript:
        raise ValueError('Transcript is empty; cannot evaluate candidate.')

    audio_metrics = analyze_audio(audio_path)
    video_metrics = analyze_video(video_path)
    transcript_metrics = _build_transcript_metrics(transcript, audio_metrics['duration'])

    try:
        llm_response = generate_evaluation(_build_prompt(transcript, audio_metrics, video_metrics))
    except ModelUnavailableError as exc:
        raise RuntimeError('LLM evaluation failed: %s' % str(exc)) from exc

    llm_scores = _normalize_llm_scores(llm_response)
    category_scores, final_score, decision, reasons, strengths, weaknesses, confidence = build_category_scores(
        llm_scores=llm_scores,
        audio_metrics=audio_metrics,
        video_metrics=video_metrics,
        transcript_metrics=transcript_metrics,
        transcript_text=transcript,
    )

    def _comment_for(*keys):
        for key in keys:
            value = llm_response.get(key)
            if isinstance(value, str) and value.strip():
                return [value.strip()]
        return []

    overall_reason = None
    for overall_key in ('reason', 'overall_reason', 'feedback', 'comment', 'comments'):
        overall_value = llm_response.get(overall_key)
        if isinstance(overall_value, str) and overall_value.strip():
            overall_reason = overall_value.strip()
            break

    llm_kpi_reasons = {
        'Communication': _comment_for('communication_reason', 'communication_comment', 'communication_feedback'),
        'Confidence': _comment_for('confidence_reason', 'confidence_comment', 'confidence_feedback'),
        'Content': _comment_for('content_reason', 'content_comment', 'content_feedback'),
        'Structure': _comment_for('structure_reason', 'structure_comment', 'structure_feedback'),
    }
    for key, value in llm_kpi_reasons.items():
        if value and value[0].strip():
            reasons[key] = value
        elif overall_reason:
            reasons[key] = [overall_reason]
        else:
            reasons[key] = [f'No KPI-specific LLM comment provided for {key}.']

    return {
        'final_score': final_score,
        'decision': decision,
        'detailed_scores': category_scores,
        'kpi_reasons': reasons,
        'strengths': strengths,
        'weaknesses': weaknesses,
        'confidence': confidence,
        'ai_feedback': _sanitize_ai_feedback(llm_response, llm_scores),
        'audio_metrics': audio_metrics,
        'video_metrics': {
            'face_presence': video_metrics.face_presence,
            'eye_contact': video_metrics.eye_contact,
            'lighting': video_metrics.lighting,
            'background_cleanliness': video_metrics.background_cleanliness,
            'camera_stability': video_metrics.camera_stability,
            'grooming': video_metrics.grooming,
            'dressing': video_metrics.dressing,
            'professionalism': video_metrics.professionalism,
            'face_confidence': video_metrics.face_confidence,
        },
        'transcript_metrics': transcript_metrics,
        'raw_llm_response': llm_response.get('raw_llm_response', None),
    }
