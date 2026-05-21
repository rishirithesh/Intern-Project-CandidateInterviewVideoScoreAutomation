import nltk
import language_tool_python
from collections import Counter

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

tool = language_tool_python.LanguageTool('en-US')


# ====================== ROLE-BASED CONFIGURATION ======================
# Easy to extend when new roles or criteria are needed
ROLE_CONFIG = {
    "general": {
        "weights": {
            "vocabulary": 0.25,
            "complexity": 0.20,
            "grammar": 0.20,
            "filler": 0.15,
            "speed": 0.20
        },
        "criteria_names": {
            "vocabulary": "Vocabulary Richness",
            "complexity": "Sentence Complexity",
            "grammar": "Grammar Accuracy",
            "filler": "Filler Word Control",
            "speed": "Speaking Speed"
        }
    },
    # Add new roles easily here in future
    "sales": {
        "weights": {
            "vocabulary": 0.20,
            "complexity": 0.15,
            "grammar": 0.15,
            "filler": 0.15,
            "speed": 0.15,
            "persuasion": 0.20   # Example new criteria
        },
        "criteria_names": {
            "vocabulary": "Vocabulary Richness",
            "complexity": "Sentence Complexity",
            "grammar": "Grammar Accuracy",
            "filler": "Filler Word Control",
            "speed": "Speaking Speed",
            "persuasion": "Persuasion Skills"
        }
    }
    # You can add "technical", "leadership", "hr", etc.
}


def evaluate_candidate(transcript: str, segments: list = None, role: str = "general") -> dict:
    """
    Evaluate candidate based on transcript.
    Supports different roles in future.
    """
    if not transcript or not transcript.strip():
        return {
            "vocabulary": 50, "complexity": 50, "grammar": 50,
            "filler": 50, "speed": 50, "final_score": 50
        }

    words = nltk.word_tokenize(transcript.lower())
    sentences = nltk.sent_tokenize(transcript)
    total_words = len(words)

    scores = {}

    # 1. Vocabulary Richness
    unique_words = len(set(words))
    scores["vocabulary"] = min(100, (unique_words / max(total_words, 1)) * 180)

    # 2. Sentence Complexity
    if sentences:
        avg_words = total_words / len(sentences)
        scores["complexity"] = min(100, avg_words * 7)
    else:
        scores["complexity"] = 50

    # 3. Grammar Accuracy
    matches = tool.check(transcript)
    scores["grammar"] = max(30, 100 - len(matches) * 2.5)

    # 4. Filler Words
    fillers = {'um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'right', 'well'}
    filler_count = sum(1 for word in words if word in fillers)
    scores["filler"] = max(20, 100 - (filler_count / max(total_words, 1)) * 900)

    # 5. Speaking Speed
    speed_score = 70
    if segments and segments[-1].get('end'):
        duration_min = segments[-1]['end'] / 60.0
        if duration_min > 0:
            wpm = total_words / duration_min
            speed_score = max(30, min(100, 100 - abs(wpm - 145) * 0.7))
    scores["speed"] = speed_score

    # ====================== FUTURE EXPANSION ======================
    # Add new criteria here based on role
    # if role == "sales":
    #     scores["persuasion"] = calculate_persuasion_score(transcript)

    # Calculate Final Score using role-specific weights
    config = ROLE_CONFIG.get(role, ROLE_CONFIG["general"])
    weights = config["weights"]

    final_score = 0
    for key, score in scores.items():
        weight = weights.get(key, 0.2)
        final_score += score * weight

    return {
        "vocabulary": round(scores["vocabulary"]),
        "complexity": round(scores["complexity"]),
        "grammar": round(scores["grammar"]),
        "filler": round(scores["filler"]),
        "speed": round(scores["speed"]),
        "final_score": round(final_score)
    }


# Optional: Helper function for future criteria
def calculate_persuasion_score(transcript: str) -> int:
    """Example of new criteria you can add later"""
    persuasion_words = ['guarantee', 'best', 'proven', 'increase', 'improve', 'success', 'client', 'revenue']
    count = sum(1 for word in transcript.lower().split() if word in persuasion_words)
    return min(100, count * 15)