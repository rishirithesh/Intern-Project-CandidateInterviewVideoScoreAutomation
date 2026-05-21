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

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        flash('No file uploaded')
        return redirect(request.url)

    file = request.files['video']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        try:
            print(f"🚀 Processing: {filename}")

            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename.rsplit('.', 1)[0] + '.wav')
            print("→ Extracting audio...")
            extract_audio(video_path, audio_path)

            print("→ Transcribing...")
            transcription = transcribe_audio(audio_path)
            transcript = transcription["text"]

            print("→ Generating summary...")
            summary = generate_summary(transcript)

            print("→ Evaluating...")
            scores = evaluate_candidate(transcript, transcription.get("segments"))

            # Cleanup
            if os.path.exists(audio_path):
                os.remove(audio_path)

            print("✅ Processing completed successfully!")
            return render_template('result.html',
                                   filename=filename,
                                   transcript=transcript,
                                   summary=summary,
                                   scores=scores,
                                   final_score=scores['final_score'])

        except Exception as e:
            print("❌ ERROR:")
            print(traceback.format_exc())
            flash(f'Processing failed: {str(e)}')
            if os.path.exists(video_path):
                os.remove(video_path)
            return redirect(url_for('index'))

    flash('Invalid file format')
    return redirect(url_for('index'))


if __name__ == '__main__':
    print("🚀 Lightweight Interview Evaluator Started!")
    print("Access: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)