# ============================================================
#  HealthSathi AI — Flask Backend for Web UI
#  Connects HealthSathi_WebUI.html to your AI models
#
#  HOW TO RUN:
#  1. Make sure mBERT model folder and healthsathi.db are ready
#  2. pip install flask flask-cors
#  3. python HealthSathi_Flask.py
#  4. Open HealthSathi_WebUI.html in Chrome
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import whisper
import sqlite3
import json
import base64
import tempfile
import os
from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

app    = Flask(__name__)
CORS(app)

# ── Load models once at startup ─────────────────────────────
print("⏳ Loading models...")
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = 'healthsathi_mbert_model'
DB_PATH   = 'healthsathi.db'

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

with open(f'{MODEL_DIR}/label_classes.json') as f:
    le_classes = json.load(f)

whisper_model = whisper.load_model("small")
conn          = sqlite3.connect(DB_PATH, check_same_thread=False)

print(f"✅ All models loaded on {device}")


# ── Helper functions ────────────────────────────────────────
TELUGU_MAP = {
    'dengue':'Dengue',
    'malaria':'Malaria',
    'covid':'COVID-19',
    'diabetes':'Diabetes',

    'డెంగ్యూ':'Dengue',
    'మలేరియా':'Malaria',
    'కోవిడ్':'COVID-19',
    'కరోనా':'COVID-19',
    'డయాబెటిస్':'Diabetes',
    'షుగర్':'Diabetes',
    'టైఫాయిడ్':'Typhoid',
    'క్షయ':'Tuberculosis',
    'రక్తపోటు':'Hypertension'
}

def predict_disease(question, top_k=1):
    print("QUESTION:", question)

    q_lower = question.lower()

    for word, disease in TELUGU_MAP.items():
        if word in q_lower:
            print("MATCHED MAP:", disease)
            return disease if top_k==1 else [(disease,1.0)]

    enc = tokenizer(question, max_length=128, padding='max_length',
                    truncation=True, return_tensors='pt')

    with torch.no_grad():
        probs = torch.softmax(
            model(enc['input_ids'].to(device),
                  enc['attention_mask'].to(device)).logits, dim=1
        ).cpu().numpy()[0]

    top_idx = probs.argsort()[-top_k:][::-1]
    results = [(le_classes[i], float(probs[i])) for i in top_idx]

    print("TOP RESULTS:", results)

    return results[0][0] if top_k==1 else results

def query_db(disease_intent, user_question=None):
    cur = conn.cursor()
    q   = str(user_question).lower() if user_question else ''
    qtype = None
    if   any(w in q for w in ['symptom','sign','feel','lakshanalu']): qtype='symptoms'
    elif any(w in q for w in ['treat','cure','medicine','drug']):      qtype='treatment'
    elif any(w in q for w in ['prevent','avoid','protect']):           qtype='prevention'
    elif any(w in q for w in ['cause','spread']):                      qtype='causes'
    elif any(w in q for w in ['test','diagnos','detect']):             qtype='diagnosis'
    else:                                                               qtype='overview'
    search = disease_intent.lower()
    if qtype:
        r = cur.execute(
            "SELECT answer FROM qa_pairs WHERE focus_area LIKE ? COLLATE NOCASE AND question_type=? LIMIT 1",
            (f'%{search}%', qtype)
        ).fetchone()
        if r: return r[0]
    r = cur.execute(
        "SELECT answer FROM qa_pairs WHERE focus_area LIKE ? COLLATE NOCASE ORDER BY RANDOM() LIMIT 1",
        (f'%{search}%',)
    ).fetchone()
    return r[0] if r else "Please consult a doctor for more information."

def text_to_speech_b64(text, lang='en'):
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tmp.close()   # IMPORTANT

        tts = gTTS(text=text[:500], lang=lang, slow=False)
        tts.save(tmp.name)

        with open(tmp.name, 'rb') as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        os.remove(tmp.name)

        return audio_b64

    except Exception as e:
        print("TTS ERROR:", e)
        return None


# ── API Routes ───────────────────────────────────────────────
@app.route('/status')
def status():
    return jsonify({'ready': True, 'device': str(device)})

@app.route('/query', methods=['POST'])
def query():
    data    = request.json
    text    = data.get('text', '')
    lang    = data.get('lang', 'en')

    # Greeting handling
    if text.lower().strip() in ['hi', 'hello', 'hey', 'good morning', 'good evening']:
        return jsonify({
            'disease': 'Greeting',
            'confidence': 1.0,
            'answer': "Hello! I'm HealthSathi AI. How can I help you today?",
            'audio_b64': text_to_speech_b64(
                "Hello! I'm HealthSathi AI. How can I help you today?",
                lang
            ),
            'top3': [('Greeting', 1.0)]
        })

    top3 = predict_disease(text, top_k=3)
    disease = top3[0][0]
    conf    = top3[0][1]
    answer  = query_db(disease, text)
    audio   = text_to_speech_b64(answer, lang)

    # Log
    try:
        conn.execute(
            "INSERT INTO chat_logs (user_input,detected_intent,answer_given,language) VALUES (?,?,?,?)",
            (text, disease, answer[:500], lang)
        )
        conn.commit()
    except: pass

    return jsonify({
        'disease'   : disease,
        'confidence': conf,
        'answer'    : answer,
        'audio_b64' : audio,
        'top3'      : top3
    })

@app.route('/voice', methods=['POST'])
def voice():
    data      = request.json
    audio_b64 = data.get('audio_b64', '')
    lang      = data.get('lang', 'en')

    # 1. FIX: Strip the browser's Data URL header before decoding
    if ',' in audio_b64:
        audio_b64 = audio_b64.split(',')[1]

    # 2. FIX: Save as .webm (Chrome's native format), not .wav
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
    try:
        tmp.write(base64.b64decode(audio_b64))
        tmp.close()

        print("Audio size:", os.path.getsize(tmp.name))
        # Transcribe
        print("LANG:", lang)
        import time
        start = time.time()

        if lang == 'te':
         result = whisper_model.transcribe(
         tmp.name,
        task='transcribe'
       )
        else:
          result = whisper_model.transcribe(
          tmp.name,
          task='transcribe',
          language='en'
    )

        print("Whisper time:", time.time() - start)
        transcript = result['text'].strip()
        print("TRANSCRIPT:", transcript)
    finally:
        # Ensure the file is always deleted to prevent storage leaks
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

    # 3. FIX: Stop mBERT from guessing if Whisper heard nothing or background noise
    if not transcript or len(transcript) < 2:
        return jsonify({
            'transcript': "[Inaudible]",
            'disease'   : "Noise Detected",
            'confidence': 0.0,
            'answer'    : "I couldn't hear that clearly. Please check your microphone and try again.",
            'audio_b64' : text_to_speech_b64("I couldn't hear that clearly. Please try again.", lang)
        })

    # Process normally if words were actually heard
    top3    = predict_disease(transcript, top_k=3)
    disease = top3[0][0]
    conf    = top3[0][1]
    answer  = query_db(disease, transcript)
    audio   = text_to_speech_b64(answer, lang)

    return jsonify({
        'transcript': transcript,
        'disease'   : disease,
        'confidence': conf,
        'answer'    : answer,
        'audio_b64' : audio
    })


if __name__ == '__main__':
    print("\n🌐 Starting HealthSathi Web Server...")
    print("📱 Open HealthSathi_WebUI.html in Chrome")
    print("🔗 API running at http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
