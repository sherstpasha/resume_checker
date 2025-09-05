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
    ssml = re.sub(r"<\s*break\s+time\s*=\s*\"([0-9]+)ms\"\s*/?\s*>",
                  r'[[pause \1 ms]]', ssml, flags=re.IGNORECASE)

    # prosody → speed/pitch
    def _prosody_to_silero(m):
        inner = m.group(2)
        rate = re.search(r'rate="([^"]+)"', m.group(1))
        pitch = re.search(r'pitch="([^"]+)"', m.group(1))
        pre, post = "", ""
        if rate:
            r = rate.group(1)
            if r in ("x-slow", "slow"): pre += "[[speed 80%]]"; post = "[[/speed]]" + post
            if r in ("fast", "x-fast"): pre += "[[speed 120%]]"; post = "[[/speed]]" + post
        if pitch:
            p = pitch.group(1)
            if p in ("low", "x-low"): pre += "[[pitch -20%]]"; post = "[[/pitch]]" + post
            if p in ("high", "x-high"): pre += "[[pitch +20%]]"; post = "[[/pitch]]" + post
        return pre + inner + post

    ssml = re.sub(r"<\s*prosody([^>]*)>(.*?)</\s*prosody\s*>",
                  _prosody_to_silero, ssml, flags=re.DOTALL|re.IGNORECASE)

    # emphasis → CAPS
    ssml = re.sub(r"<\s*emphasis\s*>(.*?)</\s*emphasis\s*>",
                  lambda m: m.group(1).upper(), ssml, flags=re.IGNORECASE)

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
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
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

def extract_text_from_doc_or_rtf(path):
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
    elif ext.endswith(".doc") or ext.endswith(".rtf"):
        return extract_text_from_doc_or_rtf(file_path)
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