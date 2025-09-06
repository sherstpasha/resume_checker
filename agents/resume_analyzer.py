import json
import requests
from utils import strip_thinking_tags, parse_json_safely


class ResumeAnalyzerAgent:
    PROMPT = """
    Ты HR-эксперт. Тебе даётся текст файла и список требований вакансии.

    СНАЧАЛА ОПРЕДЕЛИ, ПОХОЖ ЛИ ТЕКСТ НА РЕЗЮМЕ (CV) КАНДИДАТА:
    - Если НЕ похоже на резюме (например, новостная статья, диплом, тестовое, пустой файл):
    • Верни JSON со статусом "Это резюме": false,
    • Заполни "Анализ" пустыми данными (см. ниже),
    • В "Общий вывод" кратко укажи причину (почему не резюме),
    • "Средняя оценка" = 0, "Вопросы" = [].
    - Если похоже на резюме — продолжай обычный анализ.

    СТРОГО ВЕРНИ ТОЛЬКО JSON (БЕЗ КОД-БЛОКОВ, БЕЗ ТЕКСТА ВНЕ JSON) С СХЕМОЙ:

    {
    "Статус": {
        "Это резюме": true|false,
        "Причина": "краткое пояснение (почему да/нет)"
    },
    "Кандидат": {
        "ФИО": "строка или null",
        "Возраст": 0 или null,
        "Пол": "мужской|женский|не указан",
        "Телефон": "строка или null",
        "Адрес": "строка или null"
    },
    "Анализ": {
        "Оценка требований": {
        "Текст требования": {"балл": 0-100, "комментарий": "текст"},
        "Текст требования": {"балл": 0-100, "комментарий": "текст"}
        },
        "Средняя оценка": 0-100,
        "Общий вывод": "текст"
    },
    "Вопросы": [
        "Вопрос 1",
        "Вопрос 2",
        "Вопрос 3",
        "Вопрос 4"
    ]
    }

    ПРАВИЛА:
    - Ключами в "Оценка требований" должны быть ИМЕННО тексты требований (не сокращай и не переименовывай).
    - Если "Это резюме" = false, то:
    • "Кандидат" = поля как есть (если можешь извлечь ФИО — хорошо; иначе null),
    • "Оценка требований" = {}, "Средняя оценка" = 0, "Вопросы" = [],
    • В "Общий вывод" — чётко укажи, что файл не похож на резюме и почему.
    - Если данных не хватает, используй null или "не указан" (для поля "Пол").
    """


    def __init__(self, api_base_url: str, model_name: str):
        self.api_base_url = api_base_url.rstrip("/")
        self.model_name = model_name

    def _call_llm(self, prompt: str) -> str:
        payload = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}]}
        url = f"{self.api_base_url}/chat/completions"
        resp = requests.post(url, json=payload, timeout=90)
        resp.raise_for_status()
        raw_answer = resp.json()["choices"][0]["message"]["content"]
        return strip_thinking_tags(raw_answer)

    def analyze_and_questions(self, resume_text: str, job_requirements: list[str]) -> dict:
        if not resume_text.strip():
            raise ValueError("❌ Пустой текст резюме")
        if not job_requirements:
            raise ValueError("❌ Нет требований вакансии")

        prompt = f"""
            {self.PROMPT}

            Требования вакансии:
            {json.dumps(job_requirements, ensure_ascii=False, indent=2)}

            Резюме кандидата:
            {resume_text}
                """.strip()

        clean_answer = self._call_llm(prompt)

        # Главное изменение — пытаемся распарсить «с обёрткой»
        try:
            return parse_json_safely(clean_answer)
        except Exception:
            # как fallback — вернём сырой ответ для отладки
            return {"raw_response": clean_answer}