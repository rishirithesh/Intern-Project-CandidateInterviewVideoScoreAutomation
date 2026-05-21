# 🎤 Interview Video Evaluator

**An AI-Powered Offline Interview Analysis Tool**

Built as an **Internship Project** at **Agilisium Consulting** to help automate and improve the hiring process.

---

## 📋 Project Information

- **Intern**: Rishi Rithesh
- **Company**: Agilisium Consulting
- **Duration**: May 2026 – July 2026
- **Mentor**: Gaurav Yadav Sir & Srivatsan Sridharan Sir
- **Objective**: To build intelligent tech agents and innovative solutions that ease and automate the hiring/recruitment process.

---

## ✨ Features

- Upload candidate interview videos (MP4, MOV, AVI)
- Automatic audio extraction
- Accurate speech-to-text transcription (Faster-Whisper)
- AI-generated summary of responses
- Comprehensive candidate evaluation with 5 key metrics
- Final weighted score (0–100)
- Fully offline & local processing
- Clean, modern web interface

---

## 🚀 Use Cases

- Initial screening of candidates
- Communication & soft skills assessment
- Campus hiring & bulk recruitment
- Sales, Customer Success, Support, and Teaching roles
- Mock interview feedback system

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Speech-to-Text**: Faster-Whisper (optimized)
- **Summarization**: DistilBART (lightweight)
- **Video Processing**: MoviePy
- **NLP & Evaluation**: NLTK, language-tool-python
- **Frontend**: HTML, CSS, Vanilla JavaScript

---

## 📁 Project Structure

```plaintext
interview-evaluator/
├── app.py
├── config.py
├── requirements.txt
├── modules/
│   ├── audio_extractor.py
│   ├── transcriber.py
│   ├── summarizer.py
│   └── evaluator.py          # Core scoring logic
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   └── style.css
└── uploads/
