# handlers/documents.py
import os
import asyncio
from aiogram import Router, types, F
from dotenv import load_dotenv

from utils import extract_text, load_job_requirements_many, list_requirement_files
from agents.resume_analyzer import ResumeAnalyzerAgent
from db import create_candidate_pending, save_analysis_for_candidate, acquire_agent_lock, release_agent_lock


# Подтягиваем .env при локальном запуске
load_dotenv()

# --- ENV ---
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
FILES_DIR = os.getenv("FILES_DIR", "./storage")
JOB_REQUIREMENTS_DIR = os.getenv("JOB_REQUIREMENTS_DIR", "")  # папка с файлами требований
RESUME_THRASH = os.getenv("RESUME_THRASH")
RESUME_THRESHOLD = (
    int(RESUME_THRASH)
    if RESUME_THRASH is not None
    else int(os.getenv("RESUME_THRESHOLD", "75"))
)


router = Router()
agent = ResumeAnalyzerAgent(api_base_url=API_BASE_URL, model_name=MODEL_NAME)

SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".doc", ".rtf", ".odt"}


def is_supported(filename: str) -> bool:
    _, ext = os.path.splitext(filename or "")
    return ext.lower() in SUPPORTED_EXTS


@router.message(F.document)
async def on_document(message: types.Message):
    doc: types.Document = message.document

    # Получен документ

    # Проверка расширения
    if not is_supported(doc.file_name or ""):
        await message.answer(
            "Формат не поддерживается. Загрузите: " + ", ".join(sorted(SUPPORTED_EXTS))
        )
        return

    await message.answer(
        "Спасибо! Резюме получено. В ближайшее время Вам поступит обратная связь!"
    )

    # --- Сохраняем файл локально ---
    try:
        save_dir = os.path.join(FILES_DIR, str(message.chat.id))
        os.makedirs(save_dir, exist_ok=True)
        local_path = os.path.join(save_dir, doc.file_name)

        file = await message.bot.get_file(doc.file_id)
        # aiogram v3: у бота есть удобный downloader
        await message.bot.download(file, destination=local_path)
    except Exception as e:
        await message.answer(f"Не удалось сохранить файл: {e}")
        return

    # --- Извлекаем текст из файла ---
    resume_text = extract_text(local_path)
    if not resume_text.strip():
        await message.answer(
            "Не получилось извлечь текст из файла. Попробуйте другой формат."
        )
        return

    # --- Загружаем требования вакансии только из папки JOB_REQUIREMENTS_DIR ---
    paths = list_requirement_files(JOB_REQUIREMENTS_DIR)
    job_sets = load_job_requirements_many(paths)
    if not job_sets:
        await message.answer(
            "Не удалось загрузить требования вакансий. Проверьте папку JOB_REQUIREMENTS_DIR."
        )
        return

    # --- Ставит в очередь: создаём кандидата со статусом ожидания и запускаем фоновую задачу анализа ---
    try:
        candidate_id = await create_candidate_pending(
            resume_path=local_path, raw_resume_text=resume_text
        )
    except Exception as e:
        await message.answer(f"Не удалось создать запись кандидата: {e}")
        return

    async def _run_analysis_and_notify():
        try:
            # глобальная очередь агента, чтобы не было параллельных запросов
            locked = await acquire_agent_lock("screening_agent", timeout_sec=1800, poll_sec=0.5)
            if not locked:
                await message.answer("Очередь обработки занята. Попробуйте позже.")
                return
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: agent.analyze_and_questions(resume_text, job_sets)
            )
            avg_score: int | None = None
            summary = ""
            try:
                if isinstance(result, dict) and isinstance(result.get("Анализ"), dict):
                    avg_score = int(result["Анализ"].get("Средняя оценка"))
                    summary = str(result["Анализ"].get("Общий вывод", ""))
            except Exception:
                pass
            status = (result.get("Статус") or {}) if isinstance(result, dict) else {}
            is_resume = bool(status.get("Это резюме", True))

            await save_analysis_for_candidate(
                candidate_id=candidate_id,
                avg_score=avg_score if isinstance(avg_score, int) else -1,
                verdict=(
                    "positive"
                    if (isinstance(avg_score, int) and avg_score >= RESUME_THRESHOLD)
                    else "negative"
                ),
                summary=summary or "—",
                raw_json=result,
                is_resume=is_resume,
                full_name=(result.get("Кандидат", {}) or {}).get("ФИО") if isinstance(result, dict) else None,
                age=(result.get("Кандидат", {}) or {}).get("Возраст") if isinstance(result, dict) else None,
                gender=(result.get("Кандидат", {}) or {}).get("Пол") if isinstance(result, dict) else None,
                phone=(result.get("Кандидат", {}) or {}).get("Телефон") if isinstance(result, dict) else None,
                address=(result.get("Кандидат", {}) or {}).get("Адрес") if isinstance(result, dict) else None,
            )

            if isinstance(avg_score, int):
                if avg_score >= RESUME_THRESHOLD:
                    await message.answer(
                        f"Ваше резюме прошло предварительную проверку (средняя оценка: {avg_score}%). Мы свяжемся с вами."
                    )
                else:
                    feedback = summary or "Спасибо за интерес к вакансии. Сейчас соответствие недостаточно."
                    await message.answer(
                        f"Предварительная оценка ниже порога (средняя: {avg_score}%).\nОбратная связь: {feedback}"
                    )
            else:
                await message.answer("Получен нестандартный ответ от анализатора.")
        except Exception as e:
            # Ошибка анализа в фоне — уведомим кратко
            await message.answer("Ошибка анализа резюме. Попробуйте позже.")
        finally:
            try:
                await release_agent_lock("screening_agent")
            except Exception:
                pass

    asyncio.create_task(_run_analysis_and_notify())
