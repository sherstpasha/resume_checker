# handlers/documents.py
import os
from aiogram import Router, types, F
from dotenv import load_dotenv

from utils import (
    extract_text,
    load_job_requirements_many,
    parse_job_paths_env,
    list_requirement_files,
)
from agents.resume_analyzer import ResumeAnalyzerAgent
from db import create_candidate_and_analysis, log_event, create_candidate_pending, save_analysis_for_candidate


# Подтягиваем .env при локальном запуске
load_dotenv()

# --- ENV ---
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
FILES_DIR = os.getenv("FILES_DIR", "./storage")
JOB_DESCRIPTION_PATHS = os.getenv(
    "JOB_DESCRIPTION_PATHS", ""
)  # список путей через , ; или перевод строки
# Для обратной совместимости: поддержка старой переменной (одна строка, можно с запятыми)
JOB_DESCRIPTION_PATH_LEGACY = os.getenv("JOB_DESCRIPTION_PATH", "")
# Папка с требованиями (каждый файл = отдельная вакансия)
JOB_REQUIREMENTS_DIR = os.getenv("JOB_REQUIREMENTS_DIR", "")
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

    # Лог: получили документ
    await log_event(
        chat_id=message.chat.id,
        user_id=(message.from_user.id if message.from_user else None),
        username=(message.from_user.username if message.from_user else None),
        event="document_received",
        payload={
            "file_name": doc.file_name,
            "mime": doc.mime_type,
            "size": doc.file_size,
        },
    )

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
        await log_event(
            message.chat.id,
            message.from_user.id if message.from_user else None,
            message.from_user.username if message.from_user else None,
            "save_failed",
            {"error": str(e)},
        )
        await message.answer(f"Не удалось сохранить файл: {e}")
        return

    # --- Извлекаем текст из файла ---
    resume_text = extract_text(local_path)
    if not resume_text.strip():
        await log_event(
            message.chat.id,
            message.from_user.id if message.from_user else None,
            message.from_user.username if message.from_user else None,
            "extract_failed",
            {"reason": "empty_text"},
        )
        await message.answer(
            "Не получилось извлечь текст из файла. Попробуйте другой формат."
        )
        return

    # --- Загружаем требования вакансии ---
    # 1) из новой переменной JOB_DESCRIPTION_PATHS
    paths = set(parse_job_paths_env(JOB_DESCRIPTION_PATHS))
    # 2) fallback: из старой JOB_DESCRIPTION_PATH (могут быть и запятые)
    if not paths and JOB_DESCRIPTION_PATH_LEGACY:
        for p in parse_job_paths_env(JOB_DESCRIPTION_PATH_LEGACY):
            paths.add(p)
    # 3) из папки JOB_REQUIREMENTS_DIR (все файлы с поддерживаемыми расширениями)
    for p in list_requirement_files(JOB_REQUIREMENTS_DIR):
        paths.add(p)

    job_sets = load_job_requirements_many(sorted(paths))
    if not job_sets:
        await log_event(
            message.chat.id,
            message.from_user.id if message.from_user else None,
            message.from_user.username if message.from_user else None,
            "requirements_missing",
            {"paths": paths},
        )
        await message.answer(
            "Не удалось загрузить требования вакансий. "
            "Проверьте переменные JOB_DESCRIPTION_PATHS/JOB_DESCRIPTION_PATH или папку JOB_REQUIREMENTS_DIR."
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

    import asyncio
    async def _run_analysis_and_notify():
        try:
            # глобальная очередь агента, чтобы не было параллельных запросов
            from db import acquire_agent_lock, release_agent_lock
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
                    call_url = os.getenv("CALL_BASE_URL", "http://localhost:8000") + f"/call/{candidate_id}"
                    await message.answer(
                        f"Ваше резюме прошло предварительную проверку (средняя оценка: {avg_score}%).\n"
                        f"Ссылка на созвон: {call_url}"
                    )
                else:
                    feedback = summary or "Спасибо за интерес к вакансии. Сейчас соответствие недостаточно."
                    await message.answer(
                        f"Предварительная оценка ниже порога (средняя: {avg_score}%).\nОбратная связь: {feedback}"
                    )
            else:
                await message.answer("Получен нестандартный ответ от анализатора.")
        except Exception as e:
            await log_event(
                message.chat.id,
                message.from_user.id if message.from_user else None,
                message.from_user.username if message.from_user else None,
                "analyze_failed_bg",
                {"error": str(e)},
            )
        finally:
            try:
                await release_agent_lock("screening_agent")
            except Exception:
                pass

    asyncio.create_task(_run_analysis_and_notify())
