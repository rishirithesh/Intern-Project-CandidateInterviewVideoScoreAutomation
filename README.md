# Candidate Video Evaluator

An offline AI-assisted interview evaluation tool for recruiter screening.

The system extracts audio from an uploaded interview video, transcribes the answer, asks a local LLM for transcript-grounded recruiter analysis, and combines that with OpenCV-derived professional presence metrics.

## Evaluation Model

The current scoring model is intentionally conservative:

- Scores are out of 10.
- Final score is a weighted average of available KPIs.
- 6.0+ = HIRE.
- 5.0 to 5.9 = BORDERLINE.
- Below 5.0 = REJECT.
- Professional presence is evaluated only from OpenCV/MediaPipe video evidence when face and frame evidence is reliable.
- Missing LLM KPI evidence fails the evaluation instead of falling back to default scores.

## KPIs

- Communication Skills: clarity, fluency, filler usage, sentence structure, concise speaking, verbal professionalism, and technical articulation.
- Technical Skills: implementation depth, APIs/databases, debugging, architecture, deployment/devops, scalability, framework knowledge, and practical problem solving.
- Project Understanding: ownership depth, production awareness, constraints, tradeoffs, feature reasoning, and business/practical understanding.
- Professional Presence: OpenCV/MediaPipe face evidence, camera alignment, lighting, background, stability, face visibility/sharpness, clothing-region coverage, and dominant clothing color when available.
- Interview Readiness: derived from the core KPI evidence rather than assigned directly.

## Anti-Inflation Rules

- No midpoint fallback scores are used for missing model output.
- No regex or partial-output score recovery is used for malformed LLM output.
- Technical scores are penalized when answers contain buzzwords without implementation detail.
- Project scores are penalized when ownership, production, or tradeoff evidence is weak.
- Communication scores are penalized for excessive fillers, rambling, poor pace, short transcripts, and weak audio clarity.
- Video/grooming metrics do not affect technical or project scores.
- Missing OpenCV visual evidence is marked unavailable instead of being replaced with a midpoint score.

## Tech Stack

- Backend: Python, Flask
- Speech-to-text: Faster-Whisper
- Summarization: DistilBART
- Video analysis: OpenCV, MediaPipe
- Audio/video processing: MoviePy, NumPy
- LLM: local OpenAI-compatible/Ollama server

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Start your local LLM server and configure:

```bash
set LLM_SERVER_URL=http://localhost:11434
set LLM_MODEL_NAME=qwen2.5:3b
set LLM_GENERATION_TIMEOUT=180
```

Run the app:

```bash
python app.py
```

Open `http://localhost:5000`.

## Project Structure

```plaintext
interview-evaluator/
  app.py
  config.py
  requirements.txt
  modules/
    audio_analyzer.py
    audio_extractor.py
    evaluator.py
    evaluation_engine.py
    llm_client.py
    summarizer.py
    transcriber.py
    video_analyzer.py
  templates/
    index.html
    result.html
  static/
    style.css
```
