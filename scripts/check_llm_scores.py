import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules import evaluator


VALID_SAMPLE = {
    "kpis": {
        "communication_skills": {
            "score": 6,
            "evidence": ["Candidate explains the flow but uses broad phrasing."],
            "strengths": ["Mostly understandable delivery."],
            "weaknesses": ["Limited structure in the answer."],
            "reasoning": "The answer is understandable but not consistently concise or structured.",
        },
        "technical_skills": {
            "score": 5,
            "subscores": {
                "conceptual_knowledge": 5,
                "implementation_depth": 4,
                "tools_and_frameworks": 6,
                "problem_solving": 5,
            },
            "evidence": ["Mentions APIs and database but does not explain implementation details."],
            "strengths": ["Some relevant technical vocabulary."],
            "weaknesses": ["No debugging or tradeoff evidence."],
            "reasoning": "Technical depth is borderline because implementation evidence is thin.",
        },
        "project_understanding": {
            "score": 4,
            "subscores": {
                "ownership": 4,
                "feature_understanding": 5,
                "architecture_or_flow": 3,
                "practical_constraints": 3,
            },
            "evidence": ["Project ownership is described vaguely."],
            "strengths": [],
            "weaknesses": ["Ownership and production readiness are unclear."],
            "reasoning": "The project explanation sounds surface-level and lacks concrete ownership detail.",
        },
    },
    "recruiter_summary": {
        "summary": "Candidate needs manual review due to shallow project evidence.",
        "biggest_strengths": ["Basic technical vocabulary"],
        "biggest_risks": ["Weak implementation detail"],
        "confidence": "medium",
    },
}


def run():
    normalized = evaluator._normalize_llm_kpis(VALID_SAMPLE)
    print("Normalized KPI names:", list(normalized.keys()))
    for name, data in normalized.items():
        print("%s: %.1f/10" % (name, data["score"]))


if __name__ == "__main__":
    run()
