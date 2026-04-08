import os
import httpx
import asyncio
import tempfile
import zipfile
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from vosk import Model, KaldiRecognizer, SetLogLevel
import wave

SetLogLevel(-1)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ASSEMBLYAI_KEY = os.environ.get("ASSEMBLYAI_KEY")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_KEY")
ASSEMBLYAI_BASE = "https://api.assemblyai.com"

# نماذج VOSK المحملة
_vosk_models = {}
_vosk_model_lock = asyncio.Lock()

VOSK_MODELS = {
    'en': 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip',
    'ar': 'https://alphacephei.com/vosk/models/vosk-model-ar-mgb2-0.4.zip',
    'fr': 'https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip',
    'es': 'https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip',
    'de': 'https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip',
    'ru': 'https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip',
    'zh': 'https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip',
    'pt': 'https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip',
    'it': 'https://alphacephei.com/vosk/models/vosk-model-small-it-0.22.zip',
    'tr': 'https://alphacephei.com/vosk/models/vosk-model-small-tr-0.3.zip',
}


async def get_vosk_model(lang: str) -> Model:
    if lang not in VOSK_MODELS:
        lang = 'en'

    if lang in _vosk_models:
        return _vosk_models[lang]

    model_dir = f'/tmp/vosk_models/{lang}'
    os.makedirs('/tmp/vosk_models', exist_ok=True)

    if not os.path.exists(model_dir):
        url = VOSK_MODELS[lang]
        zip_path = f'/tmp/vosk_models/{lang}.zip'

        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.get(url)
            with open(zip_path, 'wb') as f:
                f.write(response.content)

        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall('/tmp/vosk_models/')

        extracted = [
            d for d in os.listdir('/tmp/vosk_models/')
            if os.path.isdir(f'/tmp/vosk_models/{d}') and d != lang
        ]
        if extracted:
            os.rename(
                f'/tmp/vosk_models/{extracted[-1]}',
                model_dir
            )

        os.remove(zip_path)

    model = Model(model_dir)
    _vosk_models[lang] = model
    return model


async def upload_file_assembly(file_bytes: bytes) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ASSEMBLYAI_BASE}/v2/upload",
            headers={"authorization": ASSEMBLYAI_KEY},
            content=file_bytes,
            timeout=120,
        )
        return response.json()["upload_url"]


async def transcribe_assembly(upload_url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ASSEMBLYAI_BASE}/v2/transcript",
            headers={"authorization": ASSEMBLYAI_KEY},
            json={
                "audio_url": upload_url,
                "language_detection": True,
                "punctuate": True,
                "speech_models": ["universal-3-pro", "universal-2"],
            },
            timeout=30,
        )
        transcript_id = response.json()["id"]

        while True:
            await asyncio.sleep(3)
            poll = await client.get(
                f"{ASSEMBLYAI_BASE}/v2/transcript/{transcript_id}",
                headers={"authorization": ASSEMBLYAI_KEY},
                timeout=30,
            )
            data = poll.json()
            if data["status"] == "completed":
                return data
            elif data["status"] == "error":
                raise Exception(f"Transcription error: {data['error']}")


async def translate_text(text: str, target_language: str) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are a professional subtitle translator. Translate the following subtitle segments to {target_language}. Each segment is separated by ---. Return ONLY the translated segments separated by ---. Keep the same number of segments. Keep translations natural and concise."
                    },
                    {"role": "user", "content": text}
                ],
                "temperature": 0.3
            }
        )
    result = response.json()
    if "choices" not in result:
        raise Exception(f"DeepSeek error: {result}")
    return result["choices"][0]["message"]["content"]


# ===========================
# مسار مدفوع - AssemblyAI
# ===========================
@app.post("/process")
async def process(
    file: UploadFile = File(...),
    target_language: str = Form("English"),
    is_premium: str = Form("false")
):
    try:
        file_bytes = await file.read()
        upload_url = await upload_file_assembly(file_bytes)
        transcript_data = await transcribe_assembly(upload_url)
        words = transcript_data.get("words", [])

        if not words:
            raise HTTPException(status_code=500, detail="No words found in audio")

        segments = []
        chunk = []
        chunk_start = None
        chunk_end = None

        for i, word in enumerate(words):
            if chunk_start is None:
                chunk_start = word["start"]
            chunk.append(word["text"])
            chunk_end = word["end"]
            if len(chunk) >= 8 or i == len(words) - 1:
                segments.append({
                    "index": len(segments) + 1,
                    "start": chunk_start,
                    "end": chunk_end,
                    "text": " ".join(chunk)
                })
                chunk = []
                chunk_start = None

        combined = "\n---\n".join([s["text"] for s in segments])
        translated_text = await translate_text(combined, target_language)
        translated_parts = [p.strip() for p in translated_text.split("\n---\n")]

        final_segments = []
        for i, seg in enumerate(segments):
            final_segments.append({
                "index": seg["index"],
                "start": seg["start"],
                "end": seg["end"],
                "original": seg["text"],
                "translated": translated_parts[i] if i < len(translated_parts) else seg["text"]
            })

        return {
            "success": True,
            "segments": final_segments,
            "detected_language": transcript_data.get("language_code", "unknown")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================
# مسار مجاني - VOSK
# ===========================
@app.post("/process-free")
async def process_free(
    file: UploadFile = File(...),
    target_language: str = Form("Arabic"),
    lang: str = Form("en"),
):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        model = await get_vosk_model(lang)
        rec = KaldiRecognizer(model, 16000)
        rec.SetWords(True)

        segments = []
        current_text = []
        segment_start = 0

        with wave.open(tmp_path, 'rb') as wf:
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get('text', '').strip()
                    if text:
                        words_result = result.get('result', [])
                        start_ms = int(words_result[0]['start'] * 1000) if words_result else segment_start
                        end_ms = int(words_result[-1]['end'] * 1000) if words_result else segment_start + 5000
                        current_text.append(text)
                        if len(' '.join(current_text).split()) >= 10:
                            segments.append({
                                'index': len(segments) + 1,
                                'start': start_ms,
                                'end': end_ms,
                                'text': ' '.join(current_text)
                            })
                            current_text = []
                            segment_start = end_ms

        final = json.loads(rec.FinalResult())
        final_text = final.get('text', '').strip()
        if final_text or current_text:
            all_text = ' '.join(current_text + ([final_text] if final_text else []))
            if all_text.strip():
                segments.append({
                    'index': len(segments) + 1,
                    'start': segment_start,
                    'end': segment_start + 5000,
                    'text': all_text.strip()
                })

        os.unlink(tmp_path)

        if not segments:
            raise HTTPException(status_code=400, detail="No speech detected")

        combined = "\n---\n".join([s["text"] for s in segments])
        translated_text = await translate_text(combined, target_language)
        translated_parts = [p.strip() for p in translated_text.split("\n---\n")]

        final_segments = []
        for i, seg in enumerate(segments):
            final_segments.append({
                "index": seg["index"],
                "start": seg["start"],
                "end": seg["end"],
                "original": seg["text"],
                "translated": translated_parts[i] if i < len(translated_parts) else seg["text"]
            })

        return {
            "success": True,
            "segments": final_segments,
            "detected_language": lang
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate-only")
async def translate_only(request: dict):
    try:
        text = request.get("text", "")
        target_language = request.get("target_language", "Arabic")
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
        translated = await translate_text(text, target_language)
        return {"success": True, "translated": translated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "flextman-api"}
