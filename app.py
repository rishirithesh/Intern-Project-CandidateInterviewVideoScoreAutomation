import logging
import os
import traceback
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from config import Config
from modules.audio_extractor import extract_audio
from modules.evaluator import evaluate_candidate
from modules.llm_client import check_model_available, ModelUnavailableError
from modules.summarizer import generate_summary
from modules.transcriber import transcribe_audio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)

app = Flask(__name__)
app.config.from_object(Config)
app.logger.setLevel(logging.INFO)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health_check():
    try:
        model_id = check_model_available()
        return {
            'status': 'healthy',
            'llm_server': Config.LLM_SERVER_URL,
            'llm_model': model_id,
        }, 200
    except ModelUnavailableError as exc:
        app.logger.error('LLM health check failed: %s', exc)
        return {
            'status': 'unhealthy',
            'llm_server': Config.LLM_SERVER_URL,
            'error': str(exc),
        }, 503


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
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.splitext(filename)[0]}.wav")
    file.save(video_path)

    try:
        app.logger.info('Processing: %s', filename)

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
            ai_feedback=scores['ai_feedback'],
            raw_llm_response=scores.get('raw_llm_response', '')
        )

    except Exception as e:
        print(traceback.format_exc())
        error_message = str(e)
        raw_response = None
        if hasattr(e, 'raw_text'):
            raw_response = getattr(e, 'raw_text')
            preview = raw_response[:300].replace('\n', ' ').replace('"', '\\"')
            error_message = f"LLM returned invalid JSON output. Preview: {preview}"
        flash(f"Processing failed: {error_message}")
        if raw_response:
            app.logger.error('Raw LLM response on failure: %s', raw_response)

        if os.path.exists(video_path):
            os.remove(video_path)

        return redirect(url_for('index'))


if __name__ == '__main__':
    print("🚀 Recruiter AI Running on http://localhost:5000")
    app.run(debug=True)