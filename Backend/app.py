import os
import base64
import tempfile
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

app = Flask(__name__)
CORS(app)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("google_api_key")
MURF_API_KEY = os.getenv("MURF_API_KEY") or os.getenv("murf_api_key")
GEMINI_MODELS = ["gemini-3.5-flash", "gemini-3.6-flash"]

client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

PROMPTS = {
    "Summary": """
You are a professional tourist guide.
Provide a high-level overview of "{place}" in {language}.

Focus on:
- The historical significance
- Why the place is famous
- Key architectural or cultural highlights

Keep the explanation concise, engaging, and easy to follow.
Avoid excessive details and dates.
Limit the response to around 200 words.

Respond ONLY in {language}.""",
    "Detailed": """
You are a professional tourist guide.
Provide a detailed and immersive explanation of "{place}" in {language}.

Cover:
- Historical background and timeline
- Architectural design and unique features
- Cultural importance and notable events
- Interesting facts and visitor insights

Explain concepts clearly and in a storytelling manner.
Include relevant details and examples to create a rich experience.
Limit the response to around 400 words.

Respond ONLY in {language}.
"""
}


def generate_speech(text, voice_id, locale):
    if not MURF_API_KEY:
        raise ValueError("MURF_API_KEY is missing. Add it to the .env file and restart the server.")

    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    url = "https://global.api.murf.ai/v1/speech/stream"
    headers = {
        "api-key": MURF_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "voice_id": voice_id,
        "text": text,
        "locale": locale,
        "model": "FALCON",
        "format": "MP3",
        "sampleRate": 24000,
        "channelType": "MONO"
    }

    response = requests.post(url, headers=headers, json=data, timeout=60)
    if response.status_code == 200:
        with open(temp_audio.name, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return temp_audio

    raise RuntimeError(f"Murf API error: {response.status_code} - {response.text[:200]}")


def generate_description(place, answer_type, language):
    if not GOOGLE_API_KEY or client is None:
        raise ValueError("GOOGLE_API_KEY is missing. Add it to the .env file and restart the server.")

    prompt = PROMPTS[answer_type].format(place=place, language=language)
    last_error = None

    for attempt, model_name in enumerate(GEMINI_MODELS):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text
        except Exception as exc:
            last_error = exc
            if attempt < len(GEMINI_MODELS) - 1:
                time.sleep(1.5 * (attempt + 1))
                continue

    raise RuntimeError(f"Gemini API error after trying {len(GEMINI_MODELS)} models: {last_error}")

@app.route("/generate-audio-guide", methods=["POST"])
def generate_audio_guide():
    try:
        data = request.get_json(silent=True) or {}
        place = data["place"]
        answer_type = data["answerType"]
        language = data["language"]
        voice_id = data["voiceId"]
        locale = data["locale"]

        text_description = generate_description(place, answer_type, language)
        audio_path = generate_speech(text_description, voice_id, locale)

        with open(audio_path.name, "rb") as f:
            audio_bytes = f.read()

        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "description": text_description,
            "audioBase64": encoded_audio
        }
    except Exception as exc:
        return jsonify({
            "error": str(exc),
            "status": "error"
        }), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
