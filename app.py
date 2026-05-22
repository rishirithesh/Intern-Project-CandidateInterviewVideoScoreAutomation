import os
import traceback
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from config import Config

from modules.audio_extractor import extract_audio
from modules.transcriber import transcribe_audio
from modules.summarizer import generate_summary
from modules.evaluator import evaluate_candidate

app = Flask(__name__)
app.config.from_object(Config)

UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():

    if 'video' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('index'))

    file = request.files['video']

    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Invalid file format')
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)

    try:
        print(f"🚀 Processing: {filename}")

        # 🎥 STEP 1: Extract audio
        audio_path = os.path.join(
            UPLOAD_FOLDER,
            filename.rsplit('.', 1)[0] + '.wav'
        )

        extract_audio(video_path, audio_path)

        # 🗣 STEP 2: Transcribe
        transcription = transcribe_audio(audio_path)
        transcript = transcription["text"]

        # 📝 STEP 3: Summary
        summary = generate_summary(transcript)

        # 🧠 STEP 4: Evaluate (AI)
        scores = evaluate_candidate(
            transcript=transcript,
            audio_path=audio_path,
            video_path=video_path
        )

        # 🧹 Cleanup audio
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return render_template(
            'result.html',
            filename=filename,
            transcript=transcript,
            summary=summary,
            scores=scores,
            final_score=scores['final_score'],
            decision=scores['decision'],
            ai_feedback=scores['ai_feedback']
        )

    except Exception as e:
        print(traceback.format_exc())

        flash(f"Processing failed: {str(e)}")

        if os.path.exists(video_path):
            os.remove(video_path)

        return redirect(url_for('index'))


if __name__ == '__main__':
    print("🚀 Recruiter AI Running on http://localhost:5000")
    app.run(debug=True)