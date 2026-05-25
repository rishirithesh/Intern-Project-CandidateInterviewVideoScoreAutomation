import os
import sys

# Ensure project root is on sys.path when running as a script
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules import evaluator


SAMPLES = [
    {
        'name': 'Well-formed ints',
        'input': {'communication': 7, 'confidence': 8, 'content': 6, 'structure': 7, 'reason': 'Clear delivery.'},
    },
    {
        'name': 'Nested numeric values',
        'input': {'communication': {'value': 9}, 'confidence': {'score': 8}, 'content': {'score': 6}, 'structure': {'x': 7}, 'reason': {'note': 'Good depth.'}},
    },
    {
        'name': 'Missing fields',
        'input': {},
    },
    {
        'name': 'String numbers and invalid types',
        'input': {'communication': '9', 'confidence': None, 'content': 4.5, 'structure': {'sub': '8'}, 'reason': 'Mostly ok.'},
    },
    {
        'name': 'Float scores',
        'input': {'communication': 7.5, 'confidence': 8.2, 'content': 6.0, 'structure': 7.9},
    },
]


def run():
    print('Testing LLM score normalization and AI feedback extraction')
    for sample in SAMPLES:
        name = sample['name']
        data = sample['input']
        normalized = evaluator._normalize_llm_scores(data)
        ai_fb = evaluator._sanitize_ai_feedback(data, normalized)
        print('\n--- %s ---' % name)
        print('Raw input:', data)
        print('Normalized scores:', normalized)
        print('AI feedback value:', ai_fb)


if __name__ == '__main__':
    run()
