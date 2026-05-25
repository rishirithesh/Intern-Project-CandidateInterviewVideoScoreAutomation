from transformers import pipeline

_summarizer = None


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        print("🔄 Loading deterministic summarization model...")
        _summarizer = pipeline(
            'summarization',
            model='sshleifer/distilbart-cnn-12-6',
            device=-1,
        )
    return _summarizer


def generate_summary(text: str) -> str:
    if not text or len(text.split()) < 50:
        return text.strip()

    try:
        pipe = get_summarizer()
        output = pipe(
            text,
            max_length=130,
            min_length=50,
            do_sample=False,
        )
        return output[0]['summary_text'].strip()
    except Exception as exc:
        print(f"Summary model fallback used: {exc}")
        sentences = [sentence.strip() for sentence in text.replace('\n', ' ').split('.') if sentence.strip()]
        return ('. '.join(sentences[:4]) + '...').strip()
