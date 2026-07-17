# ============================================================
#  HealthSathi AI — Flask Backend (Final Complete Version)
#
#  STT: Google Speech API (Telugu) + Whisper fallback
#  NLP: mBERT + Telugu keyword map
#  DB:  SQLite (15,153 Q&A pairs)
#  TTS: gTTS Telugu + English
#
#  REQUIREMENTS:
#  pip install flask flask-cors torch==2.2.2 numpy==1.26.4
#  pip install openai-whisper transformers==4.40.0
#  pip install gtts googletrans==4.0.0rc1 SpeechRecognition
#
#  FOR AUDIO CONVERSION (Telugu STT needs this):
#  Windows: winget install ffmpeg
#  OR download from https://ffmpeg.org and add to PATH
#
#  HOW TO RUN:
#  python HealthSathi_Flask.py
#  Then open HealthSathi_WebUI.html in Chrome
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
import time
import subprocess
import warnings
from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)


# ════════════════════════════════════════════════════════════
# OPTIONAL IMPORTS — graceful fallback if not installed
# ════════════════════════════════════════════════════════════
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
    print("✅ SpeechRecognition available (Google STT for Telugu)")
except ImportError:
    SR_AVAILABLE = False
    print("⚠️  SpeechRecognition not installed → run: pip install SpeechRecognition")
    print("    Will use Whisper medium for Telugu instead.")

try:
    from googletrans import Translator
    translator       = Translator()
    TRANSLATE_AVAILABLE = True
    print("✅ googletrans available")
except ImportError:
    TRANSLATE_AVAILABLE = False
    print("⚠️  googletrans not installed → run: pip install googletrans==4.0.0rc1")

# Check ffmpeg
try:
    subprocess.run(['ffmpeg', '-version'],
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL,
                   check=True)
    FFMPEG_AVAILABLE = True
    print("✅ ffmpeg available")
except Exception:
    FFMPEG_AVAILABLE = False
    print("⚠️  ffmpeg not found → Telugu Google STT may not work")
    print("    Windows: winget install ffmpeg")


# ════════════════════════════════════════════════════════════
# LOAD MODELS
# ════════════════════════════════════════════════════════════
print("\n⏳ Loading AI models...")

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = 'healthsathi_mbert_model'
DB_PATH   = 'healthsathi.db'

# mBERT
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

with open(f'{MODEL_DIR}/label_classes.json') as f:
    le_classes = json.load(f)

print(f"✅ mBERT loaded on {device} | {len(le_classes)} classes")

# Whisper — medium for better Telugu accuracy
# First run downloads ~1.5GB — subsequent runs use cache
print("⏳ Loading Whisper medium (better Telugu accuracy)...")
try:
    whisper_model = whisper.load_model("medium")
    print("✅ Whisper medium loaded")
except Exception as e:
    print(f"⚠️  Whisper medium failed ({e}) — falling back to small")
    whisper_model = whisper.load_model("small")
    print("✅ Whisper small loaded")

# SQLite DB
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur  = conn.cursor()

ALL_DISEASES = [
    row[0] for row in cur.execute(
        "SELECT DISTINCT focus_area FROM qa_pairs"
    ).fetchall() if row[0]
]

print(f"✅ DB connected | {len(ALL_DISEASES)} diseases | 15,153 Q&A pairs")
print(f"\n{'='*55}")
print(f"  All models loaded! Starting server...")
print(f"{'='*55}\n")


# ════════════════════════════════════════════════════════════
# TELUGU DISEASE MAP
# Both Telugu script + transliterated Roman letters
# ════════════════════════════════════════════════════════════
TELUGU_DISEASE_MAP = {
    # Telugu script
    'డెంగ్యూ'    : 'Dengue',
    'మలేరియా'    : 'Malaria',
    'కోవిడ్'     : 'COVID-19',
    'కరోనా'      : 'COVID-19',
    'డయాబెటిస్'  : 'Diabetes',
    'షుగర్'      : 'Diabetes',
    'టైఫాయిడ్'   : 'Typhoid',
    'క్షయ'       : 'Tuberculosis',
    'రక్తపోటు'   : 'Hypertension',
    'న్యుమోనియా' : 'Pneumonia',
    'క్యాన్సర్'  : 'Cancer',
    'కలరా'       : 'Cholera',
    'హెపటైటిస్'  : 'Hepatitis',
    'చికున్‌గున్యా': 'Chikungunya',
    # Transliterated
    'dengue'         : 'Dengue',
    'denguu'         : 'Dengue',
    'malaria'        : 'Malaria',
    'maleria'        : 'Malaria',
    'covid'          : 'COVID-19',
    'corona'         : 'COVID-19',
    'korona'         : 'COVID-19',
    'typhoid'        : 'Typhoid',
    'taiephaidu'     : 'Typhoid',
    'tb'             : 'Tuberculosis',
    'kshaya'         : 'Tuberculosis',
    'tuberculosis'   : 'Tuberculosis',
    'diabetes'       : 'Diabetes',
    'sugar'          : 'Diabetes',
    'bp'             : 'Hypertension',
    'blood pressure' : 'Hypertension',
    'raktapotu'      : 'Hypertension',
    'pneumonia'      : 'Pneumonia',
    'cancer'         : 'Cancer',
    'cholera'        : 'Cholera',
    'chikungunya'    : 'Chikungunya',
    'hepatitis'      : 'Hepatitis',
}

# Telugu question type keywords
TELUGU_QTYPE_MAP = {
    'symptoms'   : ['symptom','sign','feel','lakshanalu',
                    'లక్షణాలు','enti lakshanalu','symptoms enti'],
    'treatment'  : ['treat','cure','medicine','drug',
                    'cheppandi','chikitsa','చికిత్స',
                    'ki medicine','treatment cheppandi'],
    'prevention' : ['prevent','avoid','protect','jagratta',
                    'జాగ్రత్త','niwaarinchali','ela aapali'],
    'causes'     : ['cause','spread','enduku','ఎందుకు',
                    'ela vastundi','karanam','ela varutundi'],
    'diagnosis'  : ['test','diagnos','detect','ki test',
                    'pariksha','ela telusukovalante','telusukovadaniki'],
    'overview'   : ['what is','enti','ante enti','gurinchi',
                    'గురించి','ela untundi','explain'],
}

# Telugu intro sentences for gTTS
TELUGU_INTROS = {
    'Dengue'         : 'డెంగ్యూ జ్వరం గురించి సమాచారం. ',
    'Malaria'        : 'మలేరియా గురించి సమాచారం. ',
    'COVID-19'       : 'కోవిడ్-19 గురించి సమాచారం. ',
    'Typhoid'        : 'టైఫాయిడ్ జ్వరం గురించి సమాచారం. ',
    'Tuberculosis'   : 'క్షయ రోగం గురించి సమాచారం. ',
    'Diabetes'       : 'మధుమేహం గురించి సమాచారం. ',
    'Hypertension'   : 'రక్తపోటు గురించి సమాచారం. ',
    'Pneumonia'      : 'న్యుమోనియా గురించి సమాచారం. ',
    'Cancer'         : 'క్యాన్సర్ గురించి సమాచారం. ',
    'Cholera'        : 'కలరా గురించి సమాచారం. ',
    'Chikungunya'    : 'చికున్‌గున్యా గురించి సమాచారం. ',
    'Heart Disease'  : 'హృదయ వ్యాధి గురించి సమాచారం. ',
    'Kidney Disease' : 'మూత్రపిండాల వ్యాధి గురించి సమాచారం. ',
    'Liver Disease'  : 'కాలేయ వ్యాధి గురించి సమాచారం. ',
    'Mental Health'  : 'మానసిక ఆరోగ్యం గురించి సమాచారం. ',
}


# ════════════════════════════════════════════════════════════
# AUDIO TRANSCRIPTION
# Google Speech API for Telugu (accurate)
# Whisper as fallback
# ════════════════════════════════════════════════════════════
def convert_webm_to_wav(webm_path):
    """Convert webm audio (Chrome format) to wav for SpeechRecognition."""
    wav_path = webm_path.replace('.webm', '.wav')
    try:
        subprocess.run([
            'ffmpeg', '-i', webm_path,
            '-ar', '16000',   # 16kHz sample rate
            '-ac', '1',        # mono
            '-f', 'wav',
            wav_path,
            '-y',              # overwrite
            '-loglevel', 'quiet'
        ], check=True, timeout=30)
        return wav_path
    except Exception as e:
        print(f"  ffmpeg conversion failed: {e}")
        return None


def transcribe_google(wav_path, lang='te'):
    """
    Google Speech-to-Text API — FREE, no API key needed.
    lang='te-IN' for Telugu, 'en-IN' for English.
    Excellent accuracy for both languages.
    """
    recognizer  = sr.Recognizer()
    lang_code   = 'te-IN' if lang == 'te' else 'en-IN'

    with sr.AudioFile(wav_path) as source:
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source, duration=0.3)
        audio = recognizer.record(source)

    transcript = recognizer.recognize_google(audio, language=lang_code)
    print(f"  Google STT ({lang_code}): '{transcript}'")
    return transcript


def transcribe_whisper(audio_path, lang='te'):
    """
    Whisper transcription with optimised settings for accuracy.
    Used when Google STT fails or ffmpeg not available.
    """
    lang_code = 'te' if lang == 'te' else 'en'
    result = whisper_model.transcribe(
        audio_path,
        task        = 'transcribe',
        language    = lang_code,
        temperature = 0.0,   # deterministic — no random guessing
        beam_size   = 5,     # wider beam search = better accuracy
        best_of     = 5,     # pick best of 5 decoding attempts
    )
    transcript = result['text'].strip()
    print(f"  Whisper ({lang_code}): '{transcript}'")
    return transcript


def transcribe_audio(webm_path, lang='en'):
    """
    Main transcription function.
    Telugu → Google STT (best accuracy) → Whisper fallback
    English → Google STT → Whisper fallback
    """
    transcript = ''

    # Try Google Speech API first (better for Telugu)
    if SR_AVAILABLE and FFMPEG_AVAILABLE:
        wav_path = None
        try:
            print("  Converting webm → wav...")
            wav_path = convert_webm_to_wav(webm_path)
            if wav_path and os.path.exists(wav_path):
                transcript = transcribe_google(wav_path, lang)
                return transcript
        except sr.UnknownValueError:
            print("  Google STT could not understand audio")
        except sr.RequestError as e:
            print(f"  Google STT request failed (no internet?): {e}")
        except Exception as e:
            print(f"  Google STT error: {e}")
        finally:
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)

    # Fallback to Whisper
    print("  Using Whisper fallback...")
    try:
        transcript = transcribe_whisper(webm_path, lang)
    except Exception as e:
        print(f"  Whisper also failed: {e}")
        transcript = ''

    return transcript


# ════════════════════════════════════════════════════════════
# LANGUAGE DETECTION & TRANSLATION
# ════════════════════════════════════════════════════════════
def is_telugu_text(text):
    """Check if text contains Telugu Unicode (U+0C00–U+0C7F)."""
    return any('\u0C00' <= ch <= '\u0C7F' for ch in text)


def translate_to_english(text):
    """Translate any language to English for DB lookup."""
    if not TRANSLATE_AVAILABLE:
        return text, False
    try:
        result     = translator.translate(text, dest='en')
        translated = result.text
        print(f"  Translated: '{text}' → '{translated}'")
        return translated, True
    except Exception as e:
        print(f"  Translation error: {e}")
        return text, False


# ════════════════════════════════════════════════════════════
# DISEASE PREDICTION
# ════════════════════════════════════════════════════════════
def predict_disease_mbert(question, top_k=1):
    """mBERT classification — used as fallback."""
    enc = tokenizer(question, max_length=128, padding='max_length',
                    truncation=True, return_tensors='pt')
    with torch.no_grad():
        probs = torch.softmax(
            model(enc['input_ids'].to(device),
                  enc['attention_mask'].to(device)).logits, dim=1
        ).cpu().numpy()[0]
    top_idx = probs.argsort()[-top_k:][::-1]
    results = [(le_classes[i], float(probs[i])) for i in top_idx]
    print(f"  mBERT top3: {results[:3]}")
    return results[0][0] if top_k == 1 else results


def extract_disease(text):
    """
    Fast keyword extraction.
    Checks Telugu map first, then all DB disease names.
    """
    q = text.lower().strip()
    for word, disease in TELUGU_DISEASE_MAP.items():
        if disease and word.lower() in q:
            print(f"  Keyword match: '{word}' → {disease}")
            return disease, 1.0
    for disease in ALL_DISEASES:
        if disease and disease.lower() in q:
            print(f"  DB match: {disease}")
            return disease, 1.0
    return None, 0.0


# ════════════════════════════════════════════════════════════
# DATABASE QUERY
# ════════════════════════════════════════════════════════════
def query_db(disease_intent, user_question=None):
    """Fetch best answer from SQLite for given disease + question type."""
    cur = conn.cursor()
    q   = str(user_question or '').lower()

    # Detect question type using Telugu + English keywords
    qtype = 'overview'
    for qt, keywords in TELUGU_QTYPE_MAP.items():
        if any(w in q for w in keywords):
            qtype = qt
            break

    search = disease_intent.lower()
    print(f"  DB: '{search}' | qtype='{qtype}'")

    if qtype:
        r = cur.execute(
            "SELECT answer FROM qa_pairs "
            "WHERE focus_area LIKE ? COLLATE NOCASE "
            "AND question_type=? LIMIT 1",
            (f'%{search}%', qtype)
        ).fetchone()
        if r: return r[0]

    r = cur.execute(
        "SELECT answer FROM qa_pairs "
        "WHERE focus_area LIKE ? COLLATE NOCASE "
        "ORDER BY RANDOM() LIMIT 1",
        (f'%{search}%',)
    ).fetchone()
    return r[0] if r else "Please consult a doctor for more information."


# ════════════════════════════════════════════════════════════
# TEXT TO SPEECH
# ════════════════════════════════════════════════════════════
def text_to_speech_b64(text, lang='en', disease=None):
    """
    Convert answer to speech.
    Telugu: adds Telugu intro sentence before English content
    so gTTS starts naturally in Telugu.
    """
    try:
        if lang == 'te' and disease:
            intro    = TELUGU_INTROS.get(disease, 'ఆరోగ్య సమాచారం. ')
            tts_text = intro + text[:450]
        else:
            tts_text = text[:500]

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tmp.close()
        tts = gTTS(text=tts_text, lang=lang, slow=False)
        tts.save(tmp.name)
        with open(tmp.name, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        os.remove(tmp.name)
        return b64
    except Exception as e:
        print(f"  TTS error: {e}")
        return None


# ════════════════════════════════════════════════════════════
# CORE PROCESS FUNCTION
# Used by both /query (text) and /voice (audio) routes
# ════════════════════════════════════════════════════════════
def process_query(text, lang):
    """
    Full pipeline:
    1. Detect Telugu → translate to English
    2. Extract disease (keyword → mBERT)
    3. DB lookup with English query (accesses all 15,153 records)
    4. Generate audio response
    """
    original_text  = text
    was_translated = False
    english_query  = text

    # Step 1: Translate if Telugu
    if is_telugu_text(text):
        print(f"  Telugu script detected → translating...")
        english_query, was_translated = translate_to_english(text)
    else:
        # Check for transliterated Telugu words
        te_words = ['enti','ante','ki','lo','ni','cheppandi',
                    'lakshanalu','jvaram','ela','gurinchi',
                    'vastundi','jagratta','chikitsa','kshaya',
                    'raktapotu','denguu','maleria','korona']
        if any(w in text.lower() for w in te_words):
            print(f"  Transliterated Telugu detected → translating...")
            english_query, was_translated = translate_to_english(text)

    print(f"  Processing: '{english_query}'")

    # Step 2: Extract disease — try both translated + original
    disease, conf = extract_disease(english_query)
    if not disease:
        disease, conf = extract_disease(original_text)
    if not disease:
        # mBERT on English query
        top3    = predict_disease_mbert(english_query, top_k=3)
        disease = top3[0][0]
        conf    = top3[0][1]
    else:
        top3 = [(disease, conf)]

    # Step 3: DB lookup
    answer = query_db(disease, english_query)

    # Step 4: TTS
    audio_b64 = text_to_speech_b64(answer, lang, disease)

    print(f"  ✅ {disease} ({conf*100:.1f}%) | translated={was_translated}")

    return {
        'disease'       : disease,
        'confidence'    : conf,
        'top3'          : top3,
        'answer'        : answer,
        'audio_b64'     : audio_b64,
        'english_query' : english_query if was_translated else None,
        'was_translated': was_translated,
    }


# ════════════════════════════════════════════════════════════
# API ROUTES
# ════════════════════════════════════════════════════════════
from flask import send_file

@app.route("/")
def home():
    return send_file("HealthSathi_WebUI.html")


@app.route('/status')
def status():
    return jsonify({
        'ready'           : True,
        'device'          : str(device),
        'classes'         : len(le_classes),
        'db_records'      : 15153,
        'translate_ready' : TRANSLATE_AVAILABLE,
        'google_stt_ready': SR_AVAILABLE and FFMPEG_AVAILABLE,
        'whisper_model'   : 'medium',
    })


@app.route('/query', methods=['POST'])
def query():
    data = request.json
    text = data.get('text', '').strip()
    lang = data.get('lang', 'en')

    print(f"\n📩 TEXT QUERY: '{text}' | lang={lang}")

    if not text:
        return jsonify({'error': 'Empty text'}), 400

    # Greetings
    greet_words = ['hi','hello','hey','good morning','good evening',
                   'హలో','నమస్కారం','నమస్తే']
    if text.lower().strip() in greet_words:
        msg = ("Hello! I'm HealthSathi AI. Ask me anything about health in English or Telugu!"
               if lang == 'en' else
               "నమస్కారం! నేను HealthSathi AI. ఆరోగ్యం గురించి ఏదైనా అడగండి!")
        return jsonify({
            'disease'   : 'Greeting',
            'confidence': 1.0,
            'answer'    : msg,
            'audio_b64' : text_to_speech_b64(msg, lang),
            'top3'      : [('Greeting', 1.0)],
            'was_translated': False,
        })

    result = process_query(text, lang)

    # Log to DB
    try:
        conn.execute(
            "INSERT INTO chat_logs "
            "(user_input, detected_intent, answer_given, language) "
            "VALUES (?,?,?,?)",
            (text, result['disease'], result['answer'][:500], lang)
        )
        conn.commit()
    except Exception as e:
        print(f"  Log error: {e}")

    return jsonify(result)


@app.route('/voice', methods=['POST'])
def voice():
    data      = request.json
    audio_b64 = data.get('audio_b64', '')
    lang      = data.get('lang', 'en')

    print(f"\n🎤 VOICE QUERY | lang={lang}")

    # Strip Data URL header if present
    if ',' in audio_b64:
        audio_b64 = audio_b64.split(',')[1]

    # Save audio
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
    try:
        tmp.write(base64.b64decode(audio_b64))
        tmp.close()
        print(f"  Audio saved: {os.path.getsize(tmp.name)} bytes")

        # Transcribe — Google STT → Whisper fallback
        transcript = transcribe_audio(tmp.name, lang)
        print(f"  Final transcript: '{transcript}'")

    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

    # Empty check
    if not transcript or len(transcript.strip()) < 2:
        msg = ("I couldn't hear that clearly. Please speak again."
               if lang == 'en' else
               "నేను వినలేదు. దయచేసి మళ్ళీ చెప్పండి.")
        return jsonify({
            'transcript': '[Inaudible]',
            'disease'   : 'Unknown',
            'confidence': 0.0,
            'answer'    : msg,
            'audio_b64' : text_to_speech_b64(msg, lang),
            'is_noise'  : True,
        })

    # Repeated word noise check
    words = transcript.lower().split()
    if len(words) >= 3 and len(set(words)) == 1:
        msg = ("Background noise detected. Please speak clearly."
               if lang == 'en' else
               "నేపధ్య శబ్దం వినిపించింది. దయచేసి స్పష్టంగా మాట్లాడండి.")
        return jsonify({
            'transcript': transcript,
            'disease'   : 'Noise Detected',
            'confidence': 0.0,
            'answer'    : msg,
            'audio_b64' : text_to_speech_b64(msg, lang),
            'is_noise'  : True,
        })

    # Process through full pipeline
    result             = process_query(transcript, lang)
    result['transcript'] = transcript
    result['is_noise']   = False

    # Log to DB
    try:
        conn.execute(
            "INSERT INTO chat_logs "
            "(user_input, detected_intent, answer_given, language) "
            "VALUES (?,?,?,?)",
            (transcript, result['disease'], result['answer'][:500], lang)
        )
        conn.commit()
    except Exception as e:
        print(f"  Log error: {e}")

    return jsonify(result)


@app.route('/logs')
def logs():
    """View recent chat history."""
    rows = conn.execute(
        "SELECT user_input, detected_intent, answer_given, language, timestamp "
        "FROM chat_logs ORDER BY timestamp DESC LIMIT 20"
    ).fetchall()
    return jsonify([
        {'input': r[0], 'intent': r[1],
         'answer': r[2][:100], 'lang': r[3], 'time': r[4]}
        for r in rows
    ])


# ════════════════════════════════════════════════════════════
# RUN SERVER
# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  🌐 HealthSathi AI — Web Server")
    print("="*55)
    print(f"  Device           : {device}")
    print(f"  mBERT classes    : {len(le_classes)}")
    print(f"  DB diseases      : {len(ALL_DISEASES)}")
    print(f"  DB records       : 15,153")
    print(f"  Whisper model    : medium")
    print(f"  Google STT       : {'✅ Ready' if SR_AVAILABLE and FFMPEG_AVAILABLE else '⚠️  Not available (install SpeechRecognition + ffmpeg)'}")
    print(f"  Translation      : {'✅ Ready' if TRANSLATE_AVAILABLE else '⚠️  Not available'}")
    print()
    print("  Open  → HealthSathi_WebUI.html in Chrome")
    print("  API   → http://localhost:5000")
    print("  Logs  → http://localhost:5000/logs")
    print("  Stop  → Ctrl+C")
    print("="*55 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
