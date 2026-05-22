import nltk
import re
import requests
from modules.audio_analyzer import analyze_audio
from modules.video_analyzer import analyze_video


def llm_evaluate(transcript):

    prompt = f"""
You are a strict hiring manager.

Evaluate the candidate interview response.

Transcript:
{transcript[:700]}

Return JSON ONLY:
{{
  "communication": score (1-10),
  "confidence": score (1-10),
  "content": score (1-10),
  "structure": score (1-10),
  "reason": "short explanation"
}}
"""

    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False,
                "format": "json"
            }
        )

        import json
        return json.loads(res.json()["response"])

    except:
        return {
            "communication": 5,
            "confidence": 5,
            "content": 5,
            "structure": 5,
            "reason": "LLM failure"
        }


def evaluate_candidate(transcript, audio_path, video_path):

    words = nltk.word_tokenize(transcript.lower())
    sentences = nltk.sent_tokenize(transcript)

    audio = analyze_audio(audio_path)
    video = analyze_video(video_path)
    llm = llm_evaluate(transcript)

    filler_words = {'um', 'uh', 'like', 'you know'}
    filler_count = sum(1 for w in words if w in filler_words)

    repeated = len(re.findall(r'(\b\w+\b).*\1.*\1', transcript.lower()))

    energy = audio["energy"]
    duration = audio["duration"]
    speech_rate = audio["speech_rate"]

    face_presence = video["face_presence"]
    movement = video["movement"]

    kpi_reasons = {}

    # ================= KPI FIXED =================

    # Communication
    if filler_count > 8:
        comm_signal, reason = 4, ["Frequent hesitation and fillers"]
    elif filler_count > 4:
        comm_signal, reason = 6, ["Moderate filler usage"]
    else:
        comm_signal, reason = 8.5, ["Clear and fluent speech"]

    communication = 0.7 * llm["communication"] + 0.3 * comm_signal
    kpi_reasons["Communication Quality"] = reason

    # Confidence
    if speech_rate < 0.2:
        conf_signal, reason = 4, ["Slow and hesitant delivery"]
    elif speech_rate < 0.4:
        conf_signal, reason = 6, ["Moderate speaking pace"]
    else:
        conf_signal, reason = 8.5, ["Confident delivery"]

    confidence = 0.7 * llm["confidence"] + 0.3 * conf_signal
    kpi_reasons["Confidence"] = reason

    # Content
    content = llm["content"]
    if content > 8:
        kpi_reasons["Content Quality"] = ["Strong conceptual clarity"]
    elif content > 5:
        kpi_reasons["Content Quality"] = ["Average depth"]
    else:
        kpi_reasons["Content Quality"] = ["Weak or unclear answer"]

    # Engagement (FIXED)
    if energy < 0.15:
        engagement, reason = 4, ["Very low vocal energy"]
    elif energy < 0.3:
        engagement, reason = 6, ["Moderate vocal energy"]
    elif energy < 0.6:
        engagement, reason = 8, ["Good engagement"]
    else:
        engagement, reason = 9.5, ["Strong energetic delivery"]

    kpi_reasons["Engagement"] = reason

    # Structure
    if len(sentences) < 3:
        struct_signal, reason = 4, ["Poor structure"]
    elif len(sentences) < 6:
        struct_signal, reason = 6.5, ["Moderate structure"]
    else:
        struct_signal, reason = 8.5, ["Well structured answer"]

    structure = 0.7 * llm["structure"] + 0.3 * struct_signal
    kpi_reasons["Structure"] = reason

    # Originality
    if repeated > 4:
        originality, reason = 4, ["Highly repetitive"]
    elif repeated > 2:
        originality, reason = 6, ["Some repetition"]
    else:
        originality, reason = 8.5, ["Natural expression"]

    kpi_reasons["Originality"] = reason

    # Visual Presence
    if face_presence < 0.4:
        visual, reason = 4, ["Face not visible consistently"]
    elif face_presence < 0.7:
        visual, reason = 6.5, ["Partial visibility"]
    else:
        visual, reason = 9, ["Strong presence"]

    kpi_reasons["Visual Presence"] = reason

    # Stability
    if movement > 0.6:
        stability, reason = 4, ["Excessive movement"]
    elif movement > 0.3:
        stability, reason = 6, ["Some movement"]
    else:
        stability, reason = 8.5, ["Stable posture"]

    kpi_reasons["Stability"] = reason

    # Time
    if duration < 45:
        time_score, reason = 4, ["Too short"]
    elif duration < 60:
        time_score, reason = 6, ["Slightly short"]
    elif duration <= 180:
        time_score, reason = 9, ["Well paced"]
    else:
        time_score, reason = 6, ["Too long"]

    kpi_reasons["Time Management"] = reason

    parameters = [
        ("Communication Quality", communication, 15),
        ("Confidence", confidence, 15),
        ("Content Quality", content, 20),
        ("Engagement", engagement, 10),
        ("Structure", structure, 10),
        ("Originality", originality, 10),
        ("Visual Presence", visual, 5),
        ("Stability", stability, 5),
        ("Time Management", time_score, 10)
    ]

    total = 0
    detailed = {}

    for name, score, weight in parameters:
        score = max(1, min(10, score))
        weighted = (score / 10) * weight
        total += weighted

        detailed[name] = {
            "score": round(score, 2),
            "weight": weight
        }

    final_score = round(total / 10, 2)
    decision = "SELECT" if final_score >= 7 else "REJECT"

    return {
        "final_score": final_score,
        "detailed_scores": detailed,
        "decision": decision,
        "ai_feedback": llm["reason"],
        "kpi_reasons": kpi_reasons
    }