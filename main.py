import os
import io
import json
import base64
import asyncio
import logging
from functools import lru_cache

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from dotenv import load_dotenv
from openai import OpenAI

# For file extraction:
import PyPDF2
import docx
from PIL import Image
import pytesseract

# ----------------------------
# Logging and Environment Setup
# ----------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()  # Load env variables from .env

# Initialize OpenAI client (ensure OPENAI_API_KEY is set)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define your fine-tuned model ID for the Culvana Assistant
FINE_TUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:culvana::AvG5tWiG"

# Global variable for additional context from an uploaded file
uploaded_file_content = None

# ----------------------------
# FastAPI Application Initialization
# ----------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------------------
# Health Check Endpoint
# ----------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "openai": "connected" if client.api_key else "not configured",
            "websocket": "available"
        }
    }

# ----------------------------
# Global Exception Handler
# ----------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global Exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__}
    )

# ----------------------------
# Cached Response Function (sync version)
# ----------------------------
@lru_cache(maxsize=128)
def _sync_culvana_response(query: str, file_context: str = None) -> str:
    logger.debug(f"Generating response for query: {query} with file context: {file_context}")
    system_message = "You are an assistant for Culvana."
    if file_context:
        system_message += f" The user has uploaded a file with the following content: {file_context}"
    response = client.chat.completions.create(
        model=FINE_TUNED_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ],
        max_tokens=300,
        temperature=0.0
    )
    bot_response = response.choices[0].message.content
    logger.debug(f"Response generated: {bot_response}")
    return bot_response

# Async wrapper for the cached response function
async def cached_culvana_response(query: str, file_context: str = None) -> str:
    return await asyncio.to_thread(_sync_culvana_response, query, file_context)

# ----------------------------
# Text-to-Speech (TTS) Function
# ----------------------------
async def TTS(text: str) -> bytes:
    if not text:
        raise ValueError("Empty text provided for TTS")
    try:
        logger.debug(f"Generating TTS for text: {text}")
        def sync_tts():
            response = client.Audio.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
            return response["content"]
        tts_audio_bytes = await asyncio.to_thread(sync_tts)
        logger.debug(f"TTS audio generated, bytes: {len(tts_audio_bytes)}")
        return tts_audio_bytes
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        raise ValueError(f"Failed to generate audio: {str(e)}")

# ----------------------------
# Speech-to-Text (STT) Function
# ----------------------------
async def STT(audio_bytes: bytes) -> str:
    try:
        logger.debug("Starting STT process")
        def sync_stt():
            transcription = client.Audio.transcribe(
                model="whisper-1",
                file=io.BytesIO(audio_bytes)
            )
            return transcription["text"]
        transcribed_text = await asyncio.to_thread(sync_stt)
        logger.debug(f"Transcribed text: {transcribed_text}")
        return transcribed_text
    except Exception as e:
        logger.error(f"STT Error: {e}")
        raise ValueError(f"Failed to transcribe audio: {str(e)}")

# ----------------------------
# WebSocket Connection Manager
# ----------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Client disconnected: {websocket.client}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)
        logger.debug(f"Sent message: {message}")

manager = ConnectionManager()

# ----------------------------
# Process Incoming WebSocket Messages
# ----------------------------
async def process_message(websocket: WebSocket, message: dict):
    try:
        logger.debug(f"Processing message: {message}")
        if not isinstance(message, dict):
            raise ValueError("Invalid message format: Expected a dictionary.")
        message_type = message.get("type")
        user_message = message.get("message")
        if not message_type:
            raise ValueError("Missing 'type' in the message.")
        if not user_message:
            raise ValueError("Missing 'message' in the message.")

        if message_type == "text":
            bot_response = await cached_culvana_response(user_message, uploaded_file_content)
            await manager.send_personal_message({"type": "text", "message": bot_response}, websocket)
        elif message_type == "audio":
            bot_response = await cached_culvana_response(user_message, uploaded_file_content)
            try:
                tts_audio_bytes = await TTS(bot_response)
                response_audio_base64 = base64.b64encode(tts_audio_bytes).decode('utf-8')
                await manager.send_personal_message(
                    {"type": "audio", "message": bot_response, "audio": response_audio_base64},
                    websocket
                )
            except Exception as e:
                logger.error(f"TTS Error: {e}")
                await manager.send_personal_message({"type": "text", "message": bot_response}, websocket)
        else:
            raise ValueError(f"Unsupported message type: {message_type}")
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        await manager.send_personal_message({"type": "error", "message": str(e)}, websocket)

# ----------------------------
# WebSocket Endpoint for Chat
# ----------------------------
@app.websocket("/chat")
async def chat(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            await process_message(websocket, message)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        manager.disconnect(websocket)
        await websocket.close()

# ----------------------------
# File Upload Endpoint
# ----------------------------
@app.post("/upload_file/")
async def upload_file_endpoint(file: UploadFile = File(...)):
    """
    Upload a file to provide additional context for the assistant.
    Supported file types: txt, pdf, docx, jpg, jpeg, png.
    Maximum file size: 10MB.
    """
    global uploaded_file_content
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    try:
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File size exceeds 10MB limit.")
        file_extension = file.filename.split(".")[-1].lower()
        extracted_text = None

        if file_extension == "txt":
            extracted_text = contents.decode('utf-8', errors="ignore")
        elif file_extension == "pdf":
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
                extracted_text = ""
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {e}")
        elif file_extension == "docx":
            try:
                document = docx.Document(io.BytesIO(contents))
                extracted_text = "\n".join([para.text for para in document.paragraphs])
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to extract text from DOCX: {e}")
        elif file_extension in ["jpg", "jpeg", "png"]:
            try:
                image = Image.open(io.BytesIO(contents))
                extracted_text = pytesseract.image_to_string(image)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to extract text from image: {e}")
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Supported: txt, pdf, docx, jpg, jpeg, png."
            )
        if not extracted_text or extracted_text.strip() == "":
            raise HTTPException(status_code=400, detail="No text could be extracted from the file.")
        uploaded_file_content = extracted_text
        logger.debug(f"File processed successfully. Extracted text length: {len(extracted_text)}")
        return {"status": "success", "message": "File uploaded and processed successfully."}
    except Exception as e:
        logger.error(f"Error during file upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Audio Transcription Endpoint (STT)
# ----------------------------
@app.post("/transcribe/")
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        transcribed_text = await STT(contents)
        return {"text": transcribed_text}
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Root Endpoint (Serve index.html)
# ----------------------------
@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Error reading index.html: {e}")
        raise HTTPException(status_code=500, detail="Index file not found.")

# ----------------------------
# Run the Application
# ----------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
