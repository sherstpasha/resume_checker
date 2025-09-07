import json
import os
import requests
from utils import strip_thinking_tags
from utils import FallbackTTS
import whisper
import numpy as np


class HRInterviewerAgent:
    def __init__(self, api_base_url: str, model_name: str, analysis: dict, questions: list[str], device="cpu"):
        """
        analysis: dict с результатами ResumeAnalyzerAgent
        questions: список заранее подготовленных вопросов
        """
        self.api_base_url = api_base_url.rstrip("/")
        self.model_name = model_name
        self.analysis = analysis
        self.questions = questions
        self.dialogue_history: list[dict] = []
        self.notebook: list[str] = []  # журнал мыслей интервьюера
        self.finished = False

        # STT / TTS
        self.device = device
        # модель Whisper по умолчанию — 'medium' (лучше распознаёт, чем 'small')
        whisper_model = os.getenv("WHISPER_MODEL", "medium")
        self.stt_model = whisper.load_model(whisper_model, device=device)
        self.tts = FallbackTTS()

        # Системный промпт (обновлённый)
        system_prompt = f"""
            Ты — дружелюбный, живой интервьюер (HR). Ведёшь устный диалог с кандидатом на русском языке.
            Опирайся на анализ резюме и список вопросов ниже. Твоя цель — понять опыт, навыки и мотивацию.

            Анализ кандидата:
            {json.dumps(self.analysis, ensure_ascii=False, indent=2)}

            Список вопросов (ориентир):
            {json.dumps(self.questions, ensure_ascii=False, indent=2)}

            Стиль и правила ведения беседы:
            1) Короткие, естественные реплики (1–2 предложения), разговорный тон.
            2) Всегда задавай ОДИН конкретный вопрос за раз. Без списков вопросов.
            3) Будь дружелюбным и живым: можно короткие междометия/бэк-каналы — «угу», «понял», «спасибо», «ясно».
            4) Если кандидат задаёт вопрос по теме — кратко ответь и мягко верни разговор к следующему вопросу.
            5) Если кандидат уходит в сторону — мягко верни фокус к релевантной теме.
            6) Проси конкретику по методу STAR: ситуация → задача → действия → результат (цифры, метрики, вклад).
            7) Уточняй неясности, задавай follow-up (не более одного за раз): «А какую роль вы играли?», «Какие метрики выросли?».
            8) Избегай фантазий и неподтверждённых утверждений; опирайся на сказанное кандидатом.
            9) Термины и названия на английском пиши в русской транскрипции, чтобы TTS произнёс правильно:
            «TensorFlow» → «Тéнсорфлоу», «PyTorch» → «Пайтóрч», «Jupyter» → «Джупитер», «Git» → «Гит», «Docker» → «Докер», «SQL» → «ЭсКьюЭл», «CI/CD» → «Си/СиДи».
            10) Не используй разметку, SSML и служебные теги — только обычный текст.

            Тактика:
            - Начинай с первого вопроса из списка, далее следуй по порядку, но допускай уместные уточнения.
            - Если кандидат отвечает слишком общо — попроси пример и цифры («Можно короткий пример с результатом?»).
            - Если ответ завершён — плавно переходи к следующему логичному вопросу.
            - Завершай реплики вопросом, когда ожидаешь ответ.

            Формат ответа: только краткий текст реплики, без каких-либо JSON, тегов или метаданных.
            """
        self.dialogue_history.append({"role": "system", "content": system_prompt})

    # ====== LLM ======
    def _call_llm(self, messages: list[dict]) -> str:
        payload = {"model": self.model_name, "messages": messages, "temperature": 0.4}
        url = f"{self.api_base_url}/chat/completions"
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        raw_answer = resp.json()["choices"][0]["message"]["content"]
        return strip_thinking_tags(raw_answer)

    def reply(self, user_text: str) -> str:
        """Принимает текст кандидата, возвращает текст интервьюера"""
        if self.finished:
            return "Интервью уже завершено."

        self.dialogue_history.append({"role": "user", "content": user_text})
        assistant_text = self._call_llm(self.dialogue_history)
        self.dialogue_history.append({"role": "assistant", "content": assistant_text})

        return assistant_text

    def finish_and_report(self) -> dict:
        """Генерация финального отчёта"""
        prompt_report = """
        На основе всей беседы и анализа резюме сформируй итоговый отчёт в формате JSON:

        {
        "Критерии": {
            "Профессиональные навыки": "оценка и комментарий",
            "Коммуникация": "оценка и комментарий",
            "Мотивация и потенциал": "оценка и комментарий",
            "Софт скиллы": "оценка и комментарий"
        },
        "Общий вывод": "текстовое заключение"
        }

        Только валидный JSON, без пояснений.
        """
        messages = self.dialogue_history + [{"role": "user", "content": prompt_report}]
        report_text = self._call_llm(messages)

        try:
            report = json.loads(report_text)
        except json.JSONDecodeError:
            report = {"raw_report": report_text}

        self.finished = True
        return report

    # ====== Notebook ======
    def note(self, text: str):
        """Добавить заметку в журнал"""
        self.notebook.append(text)

    def get_notes(self) -> list[str]:
        """Вернуть все заметки"""
        return self.notebook
