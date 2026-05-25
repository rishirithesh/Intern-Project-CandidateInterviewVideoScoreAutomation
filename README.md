# Candidate Video Evaluator

An offline AI-assisted interview evaluation tool for recruiter screening.

The system extracts audio from an uploaded interview video, transcribes the answer, asks a local LLM for transcript-grounded recruiter analysis, and computes calibrated evidence-driven KPI scores out of 10.

## Evaluation Model

The current scoring model is intentionally conservative:

- Scores are out of 10.
- Final score is a weighted average of available KPIs.
- 6.0+ = HIRE.
- 5.0 to 5.9 = BORDERLINE.
- Below 5.0 = REJECT.
- Professional presence is evaluated only when video face evidence is reliable.
- Missing LLM KPI evidence fails the evaluation instead of falling back to default scores.

## KPIs

- Communication Skills: clarity, fluency, filler usage, sentence structure, concise speaking, professionalism, and technical articulation.
- Technical Skills: implementation depth, APIs/databases, debugging, architecture, deployment/devops, scalability, framework knowledge, and practical problem solving.
- Project Understanding: ownership depth, production awareness, constraints, tradeoffs, feature reasoning, and business/practical understanding.
- Professional Presence: camera/face evidence, eye contact, lighting, background, stability, and visible presentation quality when available.
- Interview Readiness: derived from the core evidence rather than assigned as a static score.

## Anti-Inflation Rules

- No midpoint fallback scores are used for missing model output.
- Technical scores are penalized when answers contain buzzwords without implementation detail.
- Project scores are penalized when ownership, production, or tradeoff evidence is weak.
- Communication scores are penalized for excessive fillers, rambling, poor pace, short transcripts, and weak audio clarity.
- Video/grooming metrics do not affect technical or project scores.

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
set LLM_MODEL_NAME=phi3:latest
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
