from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from PDFinfoExtractor import get_pdf_text, get_text_chunks, get_vector_store, user_input
import os
import shutil
import asyncio
import base64
import uuid
import tempfile
from pydub import AudioSegment
import speech_recognition as sr
import pyttsx3

app = FastAPI()

ROOT = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


# CORS (for dashboard / frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# SCHEMAS
# --------------------------------------------------
class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def serve_ui():
    index_path = ROOT / "static" / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(str(index_path))

@app.post("/upload-pdf/")
async def upload_pdf(files: List[UploadFile] = File(...)):
    os.makedirs("uploaded_pdfs", exist_ok=True)
    pdf_paths = []

    for file in files:
        file_path = f"uploaded_pdfs/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        pdf_paths.append(file_path)

    text = get_pdf_text(pdf_paths)
    chunks = get_text_chunks(text)
    get_vector_store(chunks)

    return {"status": "success", "message": "PDFs processed successfully"}


@app.post("/ask/")
async def ask_question(data: QuestionRequest):
    answer = user_input(data.question)
    return {"answer": answer}

@app.post("/voice/")
async def voice_query(voice):
    # Decode the base64 audio data
    audio_data = base64.b64decode(voice.split(",")[1])
    
    # Save to a temporary file
    temp_audio_path = f"temp_{uuid.uuid4()}.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_data)
    
    # Convert to WAV if necessary (assuming input is in a compatible format)
    audio = AudioSegment.from_file(temp_audio_path)
    wav_audio_path = temp_audio_path.replace(".wav", "_converted.wav")
    audio.export(wav_audio_path, format="wav")
    
    # Recognize speech using SpeechRecognition
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_audio_path) as source:
        audio_data = recognizer.record(source)
    
    try:
        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        text = ""
    except sr.RequestError as e:
        text = ""
    
    # Clean up temporary files
    os.remove(temp_audio_path)
    os.remove(wav_audio_path)
    
    if not text:
        return {"error": "Could not understand the audio."}
    
    # Get answer from the chatbot
    answer = user_input(text)
    
    # Convert answer to speech using pyttsx3
    tts_engine = pyttsx3.init()
    tts_audio_path = f"response_{uuid.uuid4()}.mp3"
    tts_engine.save_to_file(answer, tts_audio_path)
    tts_engine.runAndWait()
    
    # Read the generated audio file and encode it to base64
    with open(tts_audio_path, "rb") as f:
        tts_audio_data = f.read()
    
    tts_audio_base64 = base64.b64encode(tts_audio_data).decode("utf-8")
    
    # Clean up TTS audio file
    os.remove(tts_audio_path)
    
    return {"text": answer, "audio": f"data:audio/mp3;base64,{tts_audio_base64}"}

@app.get("/")
def health_check():
    return {"status": "API is running 🚀"}