import os
import threading
import time
from typing import List

import numpy as np
import requests
import sounddevice as sd
import torch
import whisper
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import tempfile
from silero_vad import load_silero_vad, get_speech_timestamps
from utils import extract_text, load_job_requirements, strip_thinking_tags
import json
from utils import FallbackTTS
from agents.resume_analyzer import ResumeAnalyzerAgent


STOP_WORDS = ["стоп", "закончим", "хватит", "достаточно", "завершим"]

# ============================ Конфиг ============================

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-2-7b-chat")
JOB_DESCRIPTION_PATH = os.getenv("JOB_DESCRIPTION_PATH")
resume_agent = ResumeAnalyzerAgent(API_BASE_URL, MODEL_NAME)

SAMPLERATE = 16000
CHUNK_SEC = 1
PAUSE_CHUNKS = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.set_float32_matmul_precision("high")

# ============================ Состояние ============================

app = FastAPI()
templates = Jinja2Templates(directory="templates")

_interview_running = False
_speaking = False
_log: List[str] = []
_lock = threading.Lock()
_system_prompt = None
_dialogue_history: List[dict] = []

# ============================ Модели ============================

vad_model = load_silero_vad(onnx=False)
_VAD_DEVICE = "cpu"

stt_model = whisper.load_model("small", device=DEVICE)

tts = FallbackTTS()

# ============================ Утилиты ============================


def log_line(text: str):
    with _lock:
        print(text)
        _log.append(text)


def check_end_dialogue(user_text: str, assistant_text: str) -> bool:
    try:
        prompt = f"""
Ты бинарный классификатор. 
Вход: реплика кандидата и ответ интервьюера.
Определи, завершилось ли интервью. 

Правила:
- Если кандидат явно сказал "стоп", "закончим", "хватит" и т.п. → end=1
- Если интервьюер сказал, что всё завершено → end=1
- Иначе → end=0

Формат ответа строго JSON:
{{"end": 0}} или {{"end": 1}}

Реплика кандидата:
{user_text}

Реплика интервьюера:
{assistant_text}
"""
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
        }
        r = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, timeout=30)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
        parsed = json.loads(strip_thinking_tags(raw))
        return parsed.get("end", 0) == 1
    except Exception as e:
        log_line(f"⚠️ Ошибка check_end_dialogue: {e}")
        return False


def ask_llm(user_text: str) -> str:
    global _system_prompt, _dialogue_history, _interview_running

    # инициализация истории с системным промптом
    if not _dialogue_history:
        _dialogue_history.append({"role": "system", "content": _system_prompt})

    # добавляем реплику пользователя
    _dialogue_history.append({"role": "user", "content": user_text})

    try:
        payload = {
            "model": MODEL_NAME,
            "messages": _dialogue_history,
            "temperature": 0.4,
        }
        r = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, timeout=60)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]

        # ✅ очищаем лишние теги (chain-of-thought, <|channel|> и пр.)
        clean = strip_thinking_tags(raw)

        # сохраняем в историю
        _dialogue_history.append({"role": "assistant", "content": clean})

        # проверка окончания интервью
        if check_end_dialogue(user_text, clean):
            log_line("📊 Модель классификатор решила: интервью закончено.")

            # генерируем финальный отчёт
            prompt_report = """
            На основе всего собеседования и анализа кандидата 
            сформируй итоговый отчёт в формате JSON:
            {
              "Критерии": {
                "Профессиональные навыки": "оценка и комментарий",
                "Коммуникация": "оценка и комментарий",
                "Мотивация и потенциал": "оценка и комментарий"
              },
              "Общий вывод": "текстовое заключение"
            }
            """
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": _system_prompt},
                    {"role": "user", "content": prompt_report},
                ],
            }
            r = requests.post(
                f"{API_BASE_URL}/chat/completions", json=payload, timeout=60
            )
            r.raise_for_status()
            report_raw = r.json()["choices"][0]["message"]["content"]

            # тоже чистим
            clean_report = strip_thinking_tags(report_raw)
            log_line("📊 Итоговый отчёт:\n" + clean_report)

            _interview_running = False
            return "Спасибо за собеседование! Мы подготовили итоговый отчёт."

        return clean

    except Exception as e:
        log_line(f"⚠️ Ошибка ask_llm: {e}")
        return "Извините, произошла ошибка при обработке ответа."


# ============================ Логика интервью ============================


def interview_loop():
    global _interview_running, _speaking

    log_line(f"🖥️ Whisper: {DEVICE} | TTS: {DEVICE} | VAD: {_VAD_DEVICE}")
    buffer = []
    silence_count = 0
    log_line("🎤 Собеседование началось. Говорите...")

    while _interview_running:
        if _speaking:
            buffer = []
            silence_count = 0
            time.sleep(0.05)
            continue

        audio = sd.rec(
            int(SAMPLERATE * CHUNK_SEC),
            samplerate=SAMPLERATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        audio = audio.flatten()

        x = torch.tensor(audio, dtype=torch.float32, device=_VAD_DEVICE)
        speech_ts = get_speech_timestamps(x, vad_model, sampling_rate=SAMPLERATE)

        if len(speech_ts) == 0:
            silence_count += 1
            if silence_count >= PAUSE_CHUNKS and len(buffer) > 0:
                chunk = np.concatenate(buffer)
                if np.max(np.abs(chunk)) > 0:
                    chunk = chunk / np.max(np.abs(chunk))

                result = stt_model.transcribe(
                    chunk, language="ru", fp16=(DEVICE == "cuda")
                )
                user_text = result["text"].strip()
                log_line(f"👤 Пользователь: {user_text}")

                try:
                    assistant_text = ask_llm(user_text)
                    log_line(f"🤖 Интервьюер: {assistant_text}")
                    _speaking = True
                    tts.speak(assistant_text)
                    _speaking = False
                except Exception as e:
                    log_line(f"⚠️ Ошибка LLM/TTS: {e}")

                buffer = []
                silence_count = 0
        else:
            silence_count = 0
            buffer.append(audio)

        time.sleep(0.01)

    log_line("⏹ Собеседование остановлено.")


# ============================ Роуты ============================


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


JOB_REQUIREMENTS = load_job_requirements(JOB_DESCRIPTION_PATH)


@app.post("/api/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    global _system_prompt
    try:
        # сохраняем файл
        tmp_dir = tempfile.gettempdir()
        save_path = os.path.join(tmp_dir, file.filename)
        with open(save_path, "wb") as f:
            f.write(await file.read())

        # извлекаем текст резюме
        resume_text = extract_text(save_path)
        if not resume_text.strip():
            return JSONResponse(
                {"error": "Не удалось извлечь текст из файла."}, status_code=400
            )

        # --- анализ резюме и генерация вопросов через агента ---
        result = resume_agent.analyze_and_questions(resume_text, JOB_REQUIREMENTS)

        analysis = result.get("Анализ", {})
        questions = result.get("Вопросы", [])

        # --- собираем системный промпт интервьюера ---
        _system_prompt = f"""
Ты играешь роль интервьюера. 
Твоя цель – вести собеседование, опираясь на анализ резюме и список заранее подготовленных вопросов.

Анализ кандидата:
{json.dumps(analysis, ensure_ascii=False, indent=2)}

Список вопросов:
{json.dumps(questions, ensure_ascii=False, indent=2)}

Правила:
- Говори только короткими репликами.
- Задавай вопросы по порядку, начиная с первого.
- ⚠️ Все английские слова (например названия библиотек или технологий) пиши в русской транскрипции,
  чтобы синтезатор речи произнёс их правильно. Пример: "TensorFlow" → "Тéнсорфлоу", "PyTorch" → "Пайтóрч".
"""

        return JSONResponse({"analysis": analysis, "questions": questions})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/start")
def api_start():
    global _interview_running, _log, _system_prompt, _dialogue_history
    if not _system_prompt:
        return JSONResponse(
            {"status": "error", "message": "Сначала загрузите резюме"}, status_code=400
        )

    with _lock:
        if not _interview_running:
            _log = []
            _dialogue_history = []
            _interview_running = True
            threading.Thread(target=interview_loop, daemon=True).start()
            return JSONResponse({"status": "started"})
    return JSONResponse({"status": "already running"})


@app.post("/api/stop")
def api_stop():
    global _interview_running
    with _lock:
        _interview_running = False
    return JSONResponse({"status": "stopped"})


@app.get("/api/speaking")
def api_speaking():
    return JSONResponse({"speaking": _speaking})


@app.get("/api/logs")
def api_logs():
    with _lock:
        return JSONResponse({"dialogue": "\n".join(_log)})


# ============================ Запуск ============================

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081)
