import os
import httpx
import tempfile
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

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_KEY")

print("Loading Whisper model...")
model = WhisperModel(
    "large-v3-turbo",
    device="cpu",
    compute_type="int8",
    download_root="/tmp/whisper_models"
)
print("Model loaded!")


@app.post("/process-free")
async def process_free(
    file: UploadFile = File(...),
    target_language: str = Form("Arabic"),
):
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".wav"
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        segments_gen, info = model.transcribe(
            tmp_path,
            language=None,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=False,
        )

        os.unlink(tmp_path)

        segments = []
        for seg in segments_gen:
            text = seg.text.strip()
            if text:
                segments.append({
                    "index": len(segments) + 1,
                    "start": int(seg.start * 1000),
                    "end": int(seg.end * 1000),
                    "text": text,
                })

        if not segments:
            raise HTTPException(
                status_code=400,
                detail="No speech detected in audio"
            )

        texts = [s["text"] for s in segments]
        combined = "\n---\n".join(texts)

        async with httpx.AsyncClient(timeout=120) as client:
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
                                f"Keep the same number of segments. "
                                f"Keep translations natural and concise."
                            ),
                        },
                        {"role": "user", "content": combined},
                    ],
                    "temperature": 0.3,
                },
            )

        result = response.json()

        if "choices" not in result:
            raise Exception(f"DeepSeek error: {result}")

        translated_text = result["choices"][0]["message"]["content"]
        translated_parts = [
            p.strip() for p in translated_text.split("\n---\n")
        ]

        final_segments = []
        for i, seg in enumerate(segments):
            final_segments.append({
                "index": seg["index"],
                "start": seg["start"],
                "end": seg["end"],
                "original": seg["text"],
                "translated": (
                    translated_parts[i]
                    if i < len(translated_parts)
                    else seg["text"]
                ),
            })

        return {
            "success": True,
            "segments": final_segments,
            "detected_language": info.language,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "flextman-free-api"}
