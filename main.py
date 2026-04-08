import os
import httpx
import tempfile
import threading
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

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


async def upload_file(file_bytes: bytes) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ASSEMBLYAI_BASE}/v2/upload",
            headers={"authorization": ASSEMBLYAI_KEY},
            content=file_bytes,
            timeout=120,
        )
        return response.json()["upload_url"]


async def transcribe(upload_url: str) -> dict:
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

        import asyncio
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


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    target_language: str = Form("English"),
    is_premium: str = Form("false")
):
    try:
        file_bytes = await file.read()
        upload_url = await upload_file(file_bytes)
        transcript_data = await transcribe(upload_url)
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

        texts = [s["text"] for s in segments]
        combined = "\n---\n".join(texts)

        translated_parts = []
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
                        {"role": "user", "content": combined}
                    ],
                    "temperature": 0.3
                }
            )

        result = response.json()
        if "choices" not in result:
            raise Exception(f"DeepSeek error: {result}")

        translated_text = result["choices"][0]["message"]["content"]
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


@app.post("/translate-only")
async def translate_only(request: dict):
    try:
        text = request.get("text", "")
        target_language = request.get("target_language", "Arabic")

        if not text:
            raise HTTPException(status_code=400, detail="No text provided")

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                f"You are a professional subtitle translator. "
                                f"Translate the following subtitle segments to {target_language}. "
                                f"Each segment is separated by ---. "
                                f"Return ONLY the translated segments separated by ---. "
                                f"Keep the same number of segments."
                            ),
                        },
                        {"role": "user", "content": text},
                    ],
                    "temperature": 0.3,
                },
            )

        result = response.json()
        if "choices" not in result:
            raise Exception(f"DeepSeek error: {result}")

        translated = result["choices"][0]["message"]["content"]
        return {"success": True, "translated": translated}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "flextman-api"}
