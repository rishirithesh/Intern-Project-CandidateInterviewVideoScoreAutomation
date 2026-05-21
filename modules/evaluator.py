import nltk
import re
from collections import Counter
import language_tool_python

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

tool = language_tool_python.LanguageTool('en-US')


def evaluate_candidate(transcript: str, segments: list = None) -> dict:
    """Evaluate candidate on multiple parameters"""
    if not transcript.strip():
        return {k: 50 for k in ["vocabulary", "complexity", "grammar", "filler", "speed", "final_score"]}

    words = nltk.word_tokenize(transcript.lower())
    sentences = nltk.sent_tokenize(transcript)
    total_words = len(words)

    # 1. Vocabulary Richness
    unique_words = len(set(words))
    vocab_score = min(100, (unique_words / max(total_words, 1)) * 200)

    # 2. Sentence Complexity
    complexity_score = 50
    if sentences:
        avg_words = total_words / len(sentences)
        complexity_score = min(100, avg_words * 7)

    # 3. Grammar Score
    matches = tool.check(transcript)
    grammar_score = max(30, 100 - len(matches) * 2.5)

    # 4. Filler Words
    fillers = {'um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'right', 'well'}
    filler_count = sum(1 for word in words if word in fillers)
    filler_score = max(20, 100 - (filler_count / max(total_words, 1)) * 900)

    # 5. Speaking Speed (WPM)
    speed_score = 70
    if segments and segments[-1].get('end'):
        duration_min = segments[-1]['end'] / 60.0
        if duration_min > 0:
            wpm = total_words / duration_min
            speed_score = max(30, min(100, 100 - abs(wpm - 145) * 0.7))

    # Final Weighted Score
    final_score = round(
        0.25 * vocab_score +
        0.20 * complexity_score +
        0.20 * grammar_score +
        0.15 * filler_score +
        0.20 * speed_score
    )

    return {
        "vocabulary": round(vocab_score),
        "complexity": round(complexity_score),
        "grammar": round(grammar_score),
        "filler": round(filler_score),
        "speed": round(speed_score),
        "final_score": final_score
    }