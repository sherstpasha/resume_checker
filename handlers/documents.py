# handlers/documents.py
import os
from aiogram import Router, types, F
from dotenv import load_dotenv

# Подтягиваем .env при локальном запуске
load_dotenv()

# --- ENV ---
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
FILES_DIR = os.getenv("FILES_DIR", "./storage")
JOB_DESCRIPTION_PATH = os.getenv("JOB_DESCRIPTION_PATH")  # путь к DOCX/PDF/TXT с требованиями
RESUME_THRASH = os.getenv("RESUME_THRASH")
RESUME_THRESHOLD = int(RESUME_THRASH) if RESUME_THRASH is not None else int(os.getenv("RESUME_THRESHOLD", "75"))

# --- Импорты утилит и агента ---
from utils import extract_text, load_job_requirements
from agents.resume_analyzer import ResumeAnalyzerAgent

# --- Простая БД на SQLite ---
from db import create_candidate_and_analysis, log_event

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
        payload={"file_name": doc.file_name, "mime": doc.mime_type, "size": doc.file_size},
    )

    # Проверка расширения
    if not is_supported(doc.file_name or ""):
        await message.answer(
            "Формат не поддерживается. Загрузите: " + ", ".join(sorted(SUPPORTED_EXTS))
        )
        return

    await message.answer("Спасибо! Резюме получено. В ближайшее время Вам поступит обратная связь!")

    # --- Сохраняем файл локально ---
    try:
        save_dir = os.path.join(FILES_DIR, str(message.chat.id))
        os.makedirs(save_dir, exist_ok=True)
        local_path = os.path.join(save_dir, doc.file_name)

        file = await message.bot.get_file(doc.file_id)
        # aiogram v3: у бота есть удобный downloader
        await message.bot.download(file, destination=local_path)
    except Exception as e:
        await log_event(message.chat.id, message.from_user.id if message.from_user else None,
                        message.from_user.username if message.from_user else None,
                        "save_failed", {"error": str(e)})
        await message.answer(f"Не удалось сохранить файл: {e}")
        return

    # --- Извлекаем текст из файла ---
    resume_text = extract_text(local_path)
    if not resume_text.strip():
        await log_event(message.chat.id, message.from_user.id if message.from_user else None,
                        message.from_user.username if message.from_user else None,
                        "extract_failed", {"reason": "empty_text"})
        await message.answer("Не получилось извлечь текст из файла. Попробуйте другой формат.")
        return

    # --- Загружаем требования вакансии ---
    job_requirements = load_job_requirements(JOB_DESCRIPTION_PATH) or []
    if not job_requirements:
        await log_event(message.chat.id, message.from_user.id if message.from_user else None,
                        message.from_user.username if message.from_user else None,
                        "requirements_missing", {"path": JOB_DESCRIPTION_PATH})
        await message.answer(
            "Не удалось загрузить требования вакансии. "
            "Проверьте переменную JOB_DESCRIPTION_PATH в .env и файл с описанием."
        )
        return

    # --- Анализируем резюме агентом ---
    try:
        result = agent.analyze_and_questions(resume_text, job_requirements)
        # Для дебага — можно оставить принт
        print(result)
    except Exception as e:
        await log_event(message.chat.id, message.from_user.id if message.from_user else None,
                        message.from_user.username if message.from_user else None,
                        "analyze_failed", {"error": str(e)})
        await message.answer(f"Ошибка анализа: {e}")
        return

    # --- Парсим среднюю оценку и общий вывод ---
    avg_score: int | None = None
    summary = ""
    try:
        if isinstance(result, dict) and isinstance(result.get("Анализ"), dict):
            avg_score = int(result["Анализ"].get("Средняя оценка"))
            summary = str(result["Анализ"].get("Общий вывод", ""))
    except Exception:
        pass

    # --- Статус "это резюме" (если есть в JSON) ---
    status = (result.get("Статус") or {}) if isinstance(result, dict) else {}
    is_resume = bool(status.get("Это резюме", True))

    # --- Сохраняем кандидата и анализ в БД ---
    try:
        await create_candidate_and_analysis(
            full_name=(result.get("Кандидат", {}) or {}).get("ФИО") if isinstance(result, dict) else None,
            age=(result.get("Кандидат", {}) or {}).get("Возраст") if isinstance(result, dict) else None,
            gender=(result.get("Кандидат", {}) or {}).get("Пол") if isinstance(result, dict) else None,
            phone=(result.get("Кандидат", {}) or {}).get("Телефон") if isinstance(result, dict) else None,
            address=(result.get("Кандидат", {}) or {}).get("Адрес") if isinstance(result, dict) else None,
            resume_path=local_path,
            raw_resume_text=resume_text,
            avg_score=avg_score if isinstance(avg_score, int) else -1,
            verdict=("positive" if (isinstance(avg_score, int) and avg_score >= RESUME_THRESHOLD) else "negative"),
            summary=summary or "—",
            raw_json=result,
            is_resume=is_resume,
        )
    except Exception as e:
        await log_event(message.chat.id, message.from_user.id if message.from_user else None,
                        message.from_user.username if message.from_user else None,
                        "db_save_failed", {"error": str(e)})
        # продолжаем отвечать пользователю даже если запись в БД не удалась

    # --- Ответ кандидату по порогу ---
    if isinstance(avg_score, int):
        if avg_score >= RESUME_THRESHOLD:
            await message.answer(
                f"Поздравляю! Ваше резюме прошло предварительную проверку (средняя оценка: {avg_score}%). "
                f"Мы свяжемся с вами для звонка/собеседования."
            )
        else:
            feedback = summary or "Спасибо за интерес к вакансии. Сейчас соответствие недостаточно."
            await message.answer(
                f"К сожалению, предварительная оценка ниже порога (средняя: {avg_score}%).\n"
                f"Обратная связь: {feedback}"
            )
    else:
        await message.answer("Получен нестандартный ответ от анализатора.")
