# interviewer_app.py
import os
import threading
import time
from typing import List

import numpy as np
import requests
import soundfile as sf
import simpleaudio as sa
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
from utils import (
    clean_and_extract_ssml,
    sanitize_ssml_for_silero,
    ssml_to_plain_text,
    extract_text,
    load_job_requirements,
    strip_thinking_tags
)
import json
STOP_WORDS = ["стоп", "закончим", "хватит", "достаточно", "завершим"]

# ============================ Конфиг ============================

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-2-7b-chat")
JOB_DESCRIPTION_PATH = os.getenv("JOB_DESCRIPTION_PATH")

SAMPLERATE   = 16000
CHUNK_SEC    = 1
PAUSE_CHUNKS = 2
TTS_SR       = 48000
TTS_SPEAKER  = "kseniya"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.set_float32_matmul_precision("high")
_dialogue_history: List[dict] = []
# ============================ Состояние ============================

app = FastAPI()
templates = Jinja2Templates(directory="templates")

_interview_running = False
_speaking = False
_log: List[str] = []
_lock = threading.Lock()
_system_prompt = None  # будет заполняться после анализа резюме

# ============================ Модели ============================

vad_model = load_silero_vad(onnx=False)
_VAD_DEVICE = "cpu"

stt_model = whisper.load_model("small", device=DEVICE)

tts_device = torch.device(DEVICE)
tts_model, _ = torch.hub.load(
    "snakers4/silero-models",
    model="silero_tts",
    language="ru",
    speaker="v4_ru"
)
tts_model.to(tts_device)

# ============================ Утилиты ============================

def log_line(text: str):
    with _lock:
        print(text)
        _log.append(text)

# interviewer_app.py
import os
import threading
import time
from typing import List

import numpy as np
import requests
import soundfile as sf
import simpleaudio as sa
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
from utils import (
    clean_and_extract_ssml,
    sanitize_ssml_for_silero,
    ssml_to_plain_text,
    extract_text,
    load_job_requirements,
    strip_thinking_tags
)
import json
STOP_WORDS = ["стоп", "закончим", "хватит", "достаточно", "завершим"]

# ============================ Конфиг ============================

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-2-7b-chat")
JOB_DESCRIPTION_PATH = os.getenv("JOB_DESCRIPTION_PATH")

SAMPLERATE   = 16000
CHUNK_SEC    = 1
PAUSE_CHUNKS = 2
TTS_SR       = 48000
TTS_SPEAKER  = "kseniya"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.set_float32_matmul_precision("high")
_dialogue_history: List[dict] = []
# ============================ Состояние ============================

app = FastAPI()
templates = Jinja2Templates(directory="templates")

_interview_running = False
_speaking = False
_log: List[str] = []
_lock = threading.Lock()
_system_prompt = None  # будет заполняться после анализа резюме

# ============================ Модели ============================

vad_model = load_silero_vad(onnx=False)
_VAD_DEVICE = "cpu"

stt_model = whisper.load_model("small", device=DEVICE)

tts_device = torch.device(DEVICE)
tts_model, _ = torch.hub.load(
    "snakers4/silero-models",
    model="silero_tts",
    language="ru",
    speaker="v4_ru"
)
tts_model.to(tts_device)

# ============================ Утилиты ============================

def log_line(text: str):
    with _lock:
        print(text)
        _log.append(text)

def check_end_dialogue(user_text: str, assistant_text: str) -> bool:
    """
    Спрашиваем у модели: закончено ли интервью?
    Ответ строго JSON: {"end": 0} или {"end": 1}
    """
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
        payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}]}
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

    if not _dialogue_history:
        _dialogue_history.append({"role": "system", "content": _system_prompt})

    _dialogue_history.append({"role": "user", "content": user_text})

    payload = {"model": MODEL_NAME, "messages": _dialogue_history, "temperature": 0.4}
    r = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, timeout=60)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]

    _dialogue_history.append({"role": "assistant", "content": raw})

    # Проверяем окончание интервью
    if check_end_dialogue(user_text, raw):
        log_line("📊 Модель классификатор решила: интервью закончено.")
        # Запрос итогового отчёта
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
        payload = {"model": MODEL_NAME, "messages": [
            {"role": "system", "content": _system_prompt},
            {"role": "user", "content": prompt_report}
        ]}
        r = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, timeout=60)
        r.raise_for_status()
        report_raw = r.json()["choices"][0]["message"]["content"]
        log_line("📊 Итоговый отчёт:\n" + report_raw)
        _interview_running = False
        return "<speak><p><s>Спасибо за собеседование! Мы подготовили итоговый отчёт.</s></p></speak>"

    return clean_and_extract_ssml(raw)

def speak_ssml(ssml_text: str, filename="tts_output.wav"):
    global _speaking
    _speaking = True
    try:
        silero_text = sanitize_ssml_for_silero(ssml_text)
        audio = tts_model.apply_tts(text=silero_text, speaker=TTS_SPEAKER, sample_rate=TTS_SR)
    except Exception as e:
        log_line(f"⚠️ SSML не принят ({e}). Fallback в текст.")
        plain = ssml_to_plain_text(ssml_text)
        audio = tts_model.apply_tts(text=plain, speaker=TTS_SPEAKER, sample_rate=TTS_SR)

    sf.write(filename, audio, TTS_SR)
    play_obj = sa.WaveObject.from_wave_file(filename).play()
    while play_obj.is_playing():
        time.sleep(0.05)
    time.sleep(0.3)
    _speaking = False

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

        audio = sd.rec(int(SAMPLERATE * CHUNK_SEC), samplerate=SAMPLERATE, channels=1, dtype="float32")
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

                result = stt_model.transcribe(chunk, language="ru", fp16=(DEVICE == "cuda"))
                user_text = result["text"].strip()
                log_line(f"👤 Пользователь: {user_text}")

                try:
                    assistant_ssml = ask_llm(user_text)
                    log_line(f"🤖 Интервьюер (SSML): {assistant_ssml}")
                    speak_ssml(assistant_ssml)
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
JOB_DESCRIPTION_PATH = os.getenv("JOB_DESCRIPTION_PATH")
JOB_REQUIREMENTS = load_job_requirements(JOB_DESCRIPTION_PATH)

@app.post("/api/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    global _system_prompt
    try:
        # сохраняем во временную папку
        tmp_dir = tempfile.gettempdir()
        save_path = os.path.join(tmp_dir, file.filename)
        with open(save_path, "wb") as f:
            f.write(await file.read())

        # извлекаем текст резюме
        resume_text = extract_text(save_path)
        if not resume_text.strip():
            return JSONResponse({"error": "Не удалось извлечь текст из файла."}, status_code=400)

        # --- 1) Анализ требований ---
        prompt_analysis = f"""
Ты HR-эксперт. У тебя есть резюме кандидата и список требований вакансии.
Оцени соответствие кандидата требованиям в формате JSON (как в инструкции ниже).

Резюме:
{resume_text}

Требования:
{json.dumps(JOB_REQUIREMENTS, ensure_ascii=False, indent=2)}

Инструкция:
Верни JSON с полями "Оценка требований", "Средняя оценка", "Общий вывод".
"""
        payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt_analysis}]}
        resp = requests.post(f"{API_BASE_URL}/chat/completions", json=payload)
        resp.raise_for_status()
        raw_analysis = resp.json()["choices"][0]["message"]["content"]
        clean_analysis = strip_thinking_tags(raw_analysis)

        # --- 2) Генерация вопросов ---
        prompt_questions = f"""
Ты HR-эксперт. На основе анализа резюме и требований придумай 2 естественных вопросов,
которые помогут уточнить слабые места кандидата и раскрыть его опыт.

Анализ:
{clean_analysis}

Формат ответа:
JSON:
{{
  "Вопросы": [
    "Вопрос 1",
    "Вопрос 2",
    "...",
    "Вопрос 7"
  ]
}}
"""
        payload_q = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt_questions}]}
        resp_q = requests.post(f"{API_BASE_URL}/chat/completions", json=payload_q)
        resp_q.raise_for_status()
        raw_q = resp_q.json()["choices"][0]["message"]["content"]
        clean_q = strip_thinking_tags(raw_q)

        # --- собираем системный промпт интервьюера ---
        _system_prompt = f"""
        Ты играешь роль интервьюера. 
        Твоя цель – вести собеседование, опираясь на анализ резюме и список заранее подготовленных вопросов.

        Анализ кандидата:
        {clean_analysis}

        Список вопросов:
        {clean_q}

        Правила:
        - Говори только короткими репликами.
        - Используй строго валидный SSML (<speak>...</speak>).
        - Вставляй паузы и эмоции с помощью <break>, <prosody>, <emphasis>.
        - Задавай вопросы по порядку, начиная с первого.
        - ⚠️ Все английские слова (например названия библиотек или технологий) пиши в русской транскрипции,
          чтобы синтезатор речи произнёс их правильно. Пример: "TensorFlow" → "Тéнсорфлоу", "PyTorch" → "Пайтóрч".
        """

        return JSONResponse({
            "analysis": clean_analysis,
            "questions": clean_q
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/start")
def api_start():
    global _interview_running, _log, _system_prompt, _dialogue_history
    if not _system_prompt:
        return JSONResponse({"status": "error", "message": "Сначала загрузите резюме"}, status_code=400)

    with _lock:
        if not _interview_running:
            _log = []
            _dialogue_history = []   # <<< очищаем историю
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



def ask_llm(user_text: str) -> str:
    global _system_prompt, _dialogue_history, _interview_running

    if not _dialogue_history:
        _dialogue_history.append({"role": "system", "content": _system_prompt})

    _dialogue_history.append({"role": "user", "content": user_text})

    payload = {"model": MODEL_NAME, "messages": _dialogue_history, "temperature": 0.4}
    r = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, timeout=60)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]

    _dialogue_history.append({"role": "assistant", "content": raw})

    # Проверяем окончание интервью
    if check_end_dialogue(user_text, raw):
        log_line("📊 Модель классификатор решила: интервью закончено.")
        # Запрос итогового отчёта
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
        payload = {"model": MODEL_NAME, "messages": [
            {"role": "system", "content": _system_prompt},
            {"role": "user", "content": prompt_report}
        ]}
        r = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, timeout=60)
        r.raise_for_status()
        report_raw = r.json()["choices"][0]["message"]["content"]
        log_line("📊 Итоговый отчёт:\n" + report_raw)
        _interview_running = False
        return "<speak><p><s>Спасибо за собеседование! Мы подготовили итоговый отчёт.</s></p></speak>"

    return clean_and_extract_ssml(raw)

def speak_ssml(ssml_text: str, filename="tts_output.wav"):
    global _speaking
    _speaking = True
    try:
        silero_text = sanitize_ssml_for_silero(ssml_text)
        audio = tts_model.apply_tts(text=silero_text, speaker=TTS_SPEAKER, sample_rate=TTS_SR)
    except Exception as e:
        log_line(f"⚠️ SSML не принят ({e}). Fallback в текст.")
        plain = ssml_to_plain_text(ssml_text)
        audio = tts_model.apply_tts(text=plain, speaker=TTS_SPEAKER, sample_rate=TTS_SR)

    sf.write(filename, audio, TTS_SR)
    play_obj = sa.WaveObject.from_wave_file(filename).play()
    while play_obj.is_playing():
        time.sleep(0.05)
    time.sleep(0.3)
    _speaking = False

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

        audio = sd.rec(int(SAMPLERATE * CHUNK_SEC), samplerate=SAMPLERATE, channels=1, dtype="float32")
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

                result = stt_model.transcribe(chunk, language="ru", fp16=(DEVICE == "cuda"))
                user_text = result["text"].strip()
                log_line(f"👤 Пользователь: {user_text}")

                try:
                    assistant_ssml = ask_llm(user_text)
                    log_line(f"🤖 Интервьюер (SSML): {assistant_ssml}")
                    speak_ssml(assistant_ssml)
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
JOB_DESCRIPTION_PATH = os.getenv("JOB_DESCRIPTION_PATH")
JOB_REQUIREMENTS = load_job_requirements(JOB_DESCRIPTION_PATH)

@app.post("/api/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    global _system_prompt
    try:
        # сохраняем во временную папку
        tmp_dir = tempfile.gettempdir()
        save_path = os.path.join(tmp_dir, file.filename)
        with open(save_path, "wb") as f:
            f.write(await file.read())

        # извлекаем текст резюме
        resume_text = extract_text(save_path)
        if not resume_text.strip():
            return JSONResponse({"error": "Не удалось извлечь текст из файла."}, status_code=400)

        # --- 1) Анализ требований ---
        prompt_analysis = f"""
Ты HR-эксперт. У тебя есть резюме кандидата и список требований вакансии.
Оцени соответствие кандидата требованиям в формате JSON (как в инструкции ниже).

Резюме:
{resume_text}

Требования:
{json.dumps(JOB_REQUIREMENTS, ensure_ascii=False, indent=2)}

Инструкция:
Верни JSON с полями "Оценка требований", "Средняя оценка", "Общий вывод".
"""
        payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt_analysis}]}
        resp = requests.post(f"{API_BASE_URL}/chat/completions", json=payload)
        resp.raise_for_status()
        raw_analysis = resp.json()["choices"][0]["message"]["content"]
        clean_analysis = strip_thinking_tags(raw_analysis)

        # --- 2) Генерация вопросов ---
        prompt_questions = f"""
Ты HR-эксперт. На основе анализа резюме и требований придумай 2 естественных вопросов,
которые помогут уточнить слабые места кандидата и раскрыть его опыт.

Анализ:
{clean_analysis}

Формат ответа:
JSON:
{{
  "Вопросы": [
    "Вопрос 1",
    "Вопрос 2",
    "...",
    "Вопрос 7"
  ]
}}
"""
        payload_q = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt_questions}]}
        resp_q = requests.post(f"{API_BASE_URL}/chat/completions", json=payload_q)
        resp_q.raise_for_status()
        raw_q = resp_q.json()["choices"][0]["message"]["content"]
        clean_q = strip_thinking_tags(raw_q)

        # --- собираем системный промпт интервьюера ---
        _system_prompt = f"""
        Ты играешь роль интервьюера. 
        Твоя цель – вести собеседование, опираясь на анализ резюме и список заранее подготовленных вопросов.

        Анализ кандидата:
        {clean_analysis}

        Список вопросов:
        {clean_q}

        Правила:
        - Говори только короткими репликами.
        - Используй строго валидный SSML (<speak>...</speak>).
        - Вставляй паузы и эмоции с помощью <break>, <prosody>, <emphasis>.
        - Задавай вопросы по порядку, начиная с первого.
        - ⚠️ Все английские слова (например названия библиотек или технологий) пиши в русской транскрипции,
          чтобы синтезатор речи произнёс их правильно. Пример: "TensorFlow" → "Тéнсорфлоу", "PyTorch" → "Пайтóрч".
        """

        return JSONResponse({
            "analysis": clean_analysis,
            "questions": clean_q
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/start")
def api_start():
    global _interview_running, _log, _system_prompt, _dialogue_history
    if not _system_prompt:
        return JSONResponse({"status": "error", "message": "Сначала загрузите резюме"}, status_code=400)

    with _lock:
        if not _interview_running:
            _log = []
            _dialogue_history = []   # <<< очищаем историю
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
