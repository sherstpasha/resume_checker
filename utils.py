# utils.py
import os
import re
import json
from docx import Document
import textract
from pypdf import PdfReader


def strip_thinking_tags(text: str, tags=None) -> str:
    final_marker = "<|channel|>final<|message|>"
    if final_marker in text:
        text = text.split(final_marker, 1)[1]
    if tags is None:
        tags = ["think", "reflect", "reason"]
    for tag in tags:
        text = re.sub(rf"<{tag}>(.|\s)*?</{tag}>", "", text, flags=re.IGNORECASE)
        text = re.sub(rf"</?{tag}>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|[^>]+\|>", "", text)
    return "\n".join(line for line in text.splitlines() if line.strip()).strip()


def clean_and_extract_ssml(text: str) -> str:
    text = strip_thinking_tags(text)
    m = re.search(r"<speak>.*?</speak>", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(0) if m else f"<speak>{text.strip()}</speak>"


def sanitize_ssml_for_silero(ssml: str) -> str:
    # нормализация
    ssml = ssml.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    ssml = ssml.replace("&nbsp;", " ")
    ssml = ssml.encode("utf-8", "ignore").decode("utf-8", "ignore")

    # убираем p/s
    ssml = re.sub(r"</?\s*p\s*>", "", ssml, flags=re.IGNORECASE)
    ssml = re.sub(r"</?\s*s\s*>", "", ssml, flags=re.IGNORECASE)

    # break → [[pause]]
    ssml = re.sub(
        r"<\s*break\s+time\s*=\s*\"([0-9]+)ms\"\s*/?\s*>",
        r"[[pause \1 ms]]",
        ssml,
        flags=re.IGNORECASE,
    )

    # prosody → speed/pitch
    def _prosody_to_silero(m):
        inner = m.group(2)
        rate = re.search(r'rate="([^"]+)"', m.group(1))
        pitch = re.search(r'pitch="([^"]+)"', m.group(1))
        pre, post = "", ""
        if rate:
            r = rate.group(1)
            if r in ("x-slow", "slow"):
                pre += "[[speed 80%]]"
                post = "[[/speed]]" + post
            if r in ("fast", "x-fast"):
                pre += "[[speed 120%]]"
                post = "[[/speed]]" + post
        if pitch:
            p = pitch.group(1)
            if p in ("low", "x-low"):
                pre += "[[pitch -20%]]"
                post = "[[/pitch]]" + post
            if p in ("high", "x-high"):
                pre += "[[pitch +20%]]"
                post = "[[/pitch]]" + post
        return pre + inner + post

    ssml = re.sub(
        r"<\s*prosody([^>]*)>(.*?)</\s*prosody\s*>",
        _prosody_to_silero,
        ssml,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # emphasis → CAPS
    ssml = re.sub(
        r"<\s*emphasis\s*>(.*?)</\s*emphasis\s*>",
        lambda m: m.group(1).upper(),
        ssml,
        flags=re.IGNORECASE,
    )

    # speak → убираем
    ssml = re.sub(r"</?\s*speak\s*>", "", ssml, flags=re.IGNORECASE)

    return ssml.strip()


def ssml_to_plain_text(ssml: str) -> str:
    ssml = re.sub(r"\[\[pause ([0-9]+) ms\]\]", r" ... ", ssml)
    ssml = re.sub(r"<[^>]+>", "", ssml)
    ssml = re.sub(r"\s+", " ", ssml).strip()
    return ssml


# ---------- Парсеры файлов ----------
def extract_text_from_pdf(path):
    text = ""
    reader = PdfReader(path)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text.strip()


def extract_text_from_docx(path):
    texts = []
    try:
        doc = Document(path)
        for p in doc.paragraphs:
            if p.text.strip():
                texts.append(p.text.strip())
        for table in doc.tables:
            for row in table.rows:
                row_text = [
                    cell.text.strip() for cell in row.cells if cell.text.strip()
                ]
                if row_text:
                    texts.append(" | ".join(row_text))
        content = "\n".join(texts).strip()
        if content:
            return content
    except Exception:
        pass
    try:
        text = textract.process(path)
        return text.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text_from_rtf(path):
    # Prefer pure-Python parser to avoid external deps
    try:
        from striprtf.striprtf import rtf_to_text
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read()
        return rtf_to_text(data) or ""
    except Exception:
        # fallback to textract if available
        try:
            text = textract.process(path)
            return text.decode("utf-8", errors="ignore")
        except Exception:
            return ""


def extract_text_from_doc(path):
    # Legacy .doc — rely on textract if available in env, else empty
    try:
        text = textract.process(path)
        return text.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_text(file_path):
    ext = file_path.lower()
    if ext.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif ext.endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif ext.endswith(".txt"):
        return extract_text_from_txt(file_path)
    elif ext.endswith(".rtf"):
        return extract_text_from_rtf(file_path)
    elif ext.endswith(".doc"):
        return extract_text_from_doc(file_path)
    else:
        return ""


# ---------- Загрузка требований ----------
def load_job_requirements(path):
    if not path or not os.path.exists(path):
        return []

    text = extract_text(path)
    lines = text.splitlines()

    requirements = []
    capture = False
    for line in lines:
        if "Требования" in line:
            capture = True
            continue
        if capture:
            if not line.strip():
                break
            requirements.append(line.strip())
    return [r for r in requirements if r]


def parse_job_paths_env(value: str | None) -> list[str]:
    import re

    if not value:
        return []
    parts = re.split(r"[,\n;]+", value)
    return [p.strip() for p in parts if p.strip()]


def load_job_requirements_many(paths: list[str]) -> list[dict]:
    """
    Возвращает список вакансий:
    [{"id": 1, "name": "<имя из файла>", "path": "<путь>", "requirements": [...]}, ...]
    Пустые/невалидные файлы игнорируются.
    """
    res = []
    for p in paths or []:
        try:
            reqs = load_job_requirements(p)
        except Exception:
            reqs = []
        if reqs:
            base = os.path.basename(p)
            name = os.path.splitext(base)[0]
            res.append(
                {"id": len(res) + 1, "name": name, "path": p, "requirements": reqs}
            )
    return res


SUPPORTED_REQ_EXTS = {".pdf", ".docx", ".txt", ".doc", ".rtf", ".odt"}


def list_requirement_files(dir_path: str | None) -> list[str]:
    """Возвращает список файлов требований в папке (отфильтрованы по расширениям)."""
    if not dir_path:
        return []
    if not os.path.isdir(dir_path):
        return []
    files = []
    for name in os.listdir(dir_path):
        full = os.path.join(dir_path, name)
        if not os.path.isfile(full):
            continue
        _, ext = os.path.splitext(name)
        if ext.lower() in SUPPORTED_REQ_EXTS:
            files.append(full)
    return sorted(files)


# tts_module.py
import asyncio
import edge_tts
import simpleaudio as sa
import tempfile
import os
import torch
import soundfile as sf
import pygame


class EdgeTTS:
    def __init__(self, voice="ru-RU-DmitryNeural", rate="+10%", pitch="+0Hz"):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch

    async def synthesize(self, text: str, out_path: str = None) -> str:
        if out_path is None:
            fd, out_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)

        communicate = edge_tts.Communicate(
            text, self.voice, rate=self.rate, pitch=self.pitch
        )
        with open(out_path, "wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
        return out_path

    def speak(self, text: str):
        path = asyncio.run(self.synthesize(text))
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        return path


class SileroTTS:
    def __init__(self, device=None, speaker="kseniya", sample_rate=48000):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.speaker = speaker
        self.sample_rate = sample_rate

        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="v4_ru",
        )
        self.model.to(self.device)

    def synthesize(self, text: str, out_path: str = None) -> str:
        if out_path is None:
            fd, out_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        audio = self.model.apply_tts(
            text=text, speaker=self.speaker, sample_rate=self.sample_rate
        )
        sf.write(out_path, audio, self.sample_rate)
        return out_path

    def speak(self, text: str):
        path = self.synthesize(text)
        wave_obj = sa.WaveObject.from_wave_file(path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
        return path


class FallbackTTS:
    """Сначала EdgeTTS, если ошибка → SileroTTS"""

    def __init__(self, **kwargs):
        self.edge = EdgeTTS(
            voice=kwargs.get("voice", "ru-RU-DmitryNeural"),
            rate=kwargs.get("rate", "+10%"),
            pitch=kwargs.get("pitch", "+0Hz"),
        )
        self.silero = SileroTTS(
            device=kwargs.get("device"),
            speaker=kwargs.get("speaker", "kseniya"),
            sample_rate=kwargs.get("sample_rate", 48000),
        )

    def speak(self, text: str):
        self.silero.speak(text)


#        try:
#            print("🎤 [TTS] EdgeTTS…")
#            return self.edge.speak(text)
#        except Exception as e:
#            print(f"⚠️ EdgeTTS упал ({e}), переключаюсь на Silero.")
#            return self.silero.speak(text)


def extract_json_block(text: str) -> str:
    """
    Возвращает строку с чистым JSON:
    - убирает обёртку ```json ... ``` или ```
    - извлекает первую { ... } или [ ... ]
    """
    if not text:
        return ""
    t = strip_thinking_tags(text).strip()

    # срезать ограды кода
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE)

    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        return m.group(0)
    m = re.search(r"\[.*\]", t, flags=re.DOTALL)
    if m:
        return m.group(0)
    return t.strip()


def parse_json_safely(text: str):
    """
    Пытается распарсить JSON-ответ модели, даже если он в код-блоке.
    Возвращает dict/list или бросает ValueError.
    """
    payload = extract_json_block(text)
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decode failed: {e}\nExtracted: {payload[:500]}")


def parse_job_paths_env(value: str | None) -> list[str]:
    import re

    if not value:
        return []
    parts = re.split(r"[,\n;]+", value)
    return [p.strip() for p in parts if p.strip()]


def load_job_requirements_many(paths: list[str]) -> list[dict]:
    """
    Возвращает список вакансий:
    [{"id": 1, "name": "<имя из файла>", "path": "<путь>", "requirements": [...]}, ...]
    Пустые/невалидные файлы игнорируются.
    """
    res = []
    for p in paths or []:
        try:
            reqs = load_job_requirements(p)
        except Exception:
            reqs = []
        if reqs:
            base = os.path.basename(p)
            name = os.path.splitext(base)[0]
            res.append(
                {"id": len(res) + 1, "name": name, "path": p, "requirements": reqs}
            )
    return res
