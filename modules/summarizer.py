from transformers import pipeline

summarizer = None

def get_summarizer():
    global summarizer
    if summarizer is None:
        print("🔄 Loading lightweight summarization model...")
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",   # ~300MB instead of 1.6GB
            device=-1
        )
    return summarizer


def generate_summary(text: str) -> str:
    """Lightweight summary"""
    if len(text.split()) < 50:
        return text
    
    try:
        pipe = get_summarizer()
        summary = pipe(
            text,
            max_length=130,
            min_length=50,
            do_sample=False
        )[0]['summary_text']
        return summary
    except Exception as e:
        print(f"Summary fallback used: {e}")
        # Simple fallback
        sentences = text.split('. ')
        return '. '.join(sentences[:4]) + "..."