# run_interview_cli.py
import os
import time
import json
import torch
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from silero_vad import load_silero_vad, get_speech_timestamps
from utils import extract_text, load_job_requirements
from agents.resume_analyzer import ResumeAnalyzerAgent
from agents.interviewer import HRInterviewerAgent

# ============================ КОНФИГ ============================

# Укажи путь к файлу резюме:
RESUME_PATH = r"C:\Users\pasha\Downloads\Telegram Desktop\ML_Engineer_Шерстнев_Павел (2).pdf"  # <<< замените на свой путь

STOP_WORDS = ["стоп", "закончим", "хватит", "достаточно", "завершим"]

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-2-7b-chat")
JOB_DESCRIPTION_PATH = os.getenv("JOB_DESCRIPTION_PATH")

SAMPLERATE = 16000
CHUNK_SEC = 1
PAUSE_CHUNKS = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.set_float32_matmul_precision("high")

# ============================ ИНИЦИАЛИЗАЦИЯ ============================
print("⏳ Инициализация VAD...")
vad_model = load_silero_vad(onnx=False)
_VAD_DEVICE = "cpu"


def vad_listen_loop(interviewer: HRInterviewerAgent):
    """Слушаем микрофон кусками, отслеживаем паузы VAD,
    при паузе — склеиваем и распознаём через interviewer.stt_model."""
    print(f"🖥️ Whisper device: {DEVICE} | VAD: {_VAD_DEVICE}")
    print("🎤 Собеседование началось. Говорите... (скажите «стоп» чтобы завершить)")
    buffer = []
    silence_count = 0

    while not interviewer.finished:
        # записываем очередной фрейм
        audio = sd.rec(
            int(SAMPLERATE * CHUNK_SEC),
            samplerate=SAMPLERATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        audio = audio.flatten()

        # проверяем VAD
        x = torch.tensor(audio, dtype=torch.float32, device=_VAD_DEVICE)
        speech_ts = get_speech_timestamps(x, vad_model, sampling_rate=SAMPLERATE)

        if len(speech_ts) == 0:
            silence_count += 1
            if silence_count >= PAUSE_CHUNKS and len(buffer) > 0:
                # склеиваем накопленное
                chunk = np.concatenate(buffer)
                buffer = []
                silence_count = 0

                if np.max(np.abs(chunk)) > 0:
                    chunk = chunk / np.max(np.abs(chunk))

                # распознаём через STT из агента (у него уже есть stt_model)
                result = interviewer.stt_model.transcribe(
                    chunk, language="ru", fp16=(DEVICE == "cuda")
                )
                user_text = result["text"].strip()
                if not user_text:
                    continue

                print(f"👤 Кандидат: {user_text}")

                # быстрый выход по стоп-словам
                if any(sw in user_text.lower() for sw in STOP_WORDS):
                    break

                # ответ интервьюера (LLM внутри агента)
                reply = interviewer.reply(user_text)
                print(f"🤖 Интервьюер: {reply}")

                # озвучиваем ответ агентским TTS
                interviewer.tts.speak(reply)
        else:
            silence_count = 0
            buffer.append(audio)

        time.sleep(0.01)

    print("⏹ Собеседование остановлено.")


def main():
    # 1) проверяем резюме и вытаскиваем текст
    if not RESUME_PATH or not os.path.exists(RESUME_PATH):
        print(f"❌ RESUME_PATH не найден: {RESUME_PATH}")
        return

    print("📂 Загружаем резюме:", RESUME_PATH)
    resume_text = extract_text(RESUME_PATH)
    if not resume_text.strip():
        print("❌ Не удалось извлечь текст из файла.")
        return

    # 2) требования
    job_requirements = load_job_requirements(JOB_DESCRIPTION_PATH)

    # 3) анализ + вопросы (единым промптом)
    print("🔎 Анализ резюме...")
    analyzer = ResumeAnalyzerAgent(API_BASE_URL, MODEL_NAME)
    result = analyzer.analyze_and_questions(resume_text, job_requirements)

    analysis = result.get("Анализ", {})
    questions = result.get("Вопросы", [])

    print("\n📊 Результаты анализа:")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    print("\n❓ Вопросы/темы для интервью:")
    for q in questions:
        print("-", q)

    # 4) создаём HR-интервьюера (вся LLM-логика внутри класса)
    interviewer = HRInterviewerAgent(
        api_base_url=API_BASE_URL,
        model_name=MODEL_NAME,
        analysis=analysis,
        questions=questions,
        device=DEVICE,
    )

    # 5) голосовой цикл с VAD → Whisper → reply → TTS
    vad_listen_loop(interviewer)

    # 6) финальный отчёт
    print("\n📑 Формируем итоговый отчёт...")
    report = interviewer.finish_and_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # 7) заметки интервьюера (если будете их заполнять interviewer.note(...))
    notes = interviewer.get_notes()
    if notes:
        print("\n📝 Заметки интервьюера:")
        for n in notes:
            print("-", n)


if __name__ == "__main__":
    main()
