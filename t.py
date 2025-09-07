import os
import time
import json
import signal
import argparse

import numpy as np
import sounddevice as sd
import torch
import whisper
from dotenv import load_dotenv

from silero_vad import load_silero_vad, get_speech_timestamps


STOP_WORDS = ["стоп", "закончим", "хватит", "достаточно", "завершим"]


def main():
    parser = argparse.ArgumentParser(
        description="Standalone mic → VAD → Whisper tester (CLI)"
    )
    parser.add_argument(
        "--model",
        default=os.getenv("WHISPER_MODEL", "medium"),
        help="whisper model (tiny/base/small/medium/large)",
    )
    parser.add_argument(
        "--device", default=os.getenv("STT_DEVICE", "auto"), help="auto|cuda|cpu"
    )
    parser.add_argument(
        "--samplerate", type=int, default=int(os.getenv("SAMPLERATE", "16000"))
    )
    parser.add_argument(
        "--chunk_sec", type=float, default=float(os.getenv("CHUNK_SEC", "1"))
    )
    parser.add_argument(
        "--pause_chunks", type=int, default=int(os.getenv("PAUSE_CHUNKS", "2"))
    )
    parser.add_argument(
        "--rms_gate",
        type=float,
        default=float(os.getenv("RMS_GATE", "0.03")),
        help="RMS threshold for fallback VAD",
    )
    parser.add_argument(
        "--device_index", type=int, default=None, help="sounddevice input device index"
    )
    parser.add_argument(
        "--list_devices", action="store_true", help="print sound devices and exit"
    )
    args = parser.parse_args()

    load_dotenv()

    if args.list_devices:
        print(sd.query_devices())
        return

    # Resolve torch device
    if args.device.lower() == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device.lower()
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
    if device == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print("⏳ Инициализация VAD…")
    vad_model = load_silero_vad(onnx=False)
    _VAD_DEVICE = "cpu"
    print("✅ VAD загружен (устройство: CPU)")

    print(f"⏳ Загрузка Whisper ({args.model}) на {device}…")
    stt_model = whisper.load_model(args.model, device=device)
    print("✅ Whisper загружен")

    samplerate = int(args.samplerate)
    chunk_sec = float(args.chunk_sec)
    pause_chunks = int(args.pause_chunks)
    rms_gate = float(args.rms_gate)

    print(f"🖥️ Whisper device: {device} | VAD: {_VAD_DEVICE}")
    print("🎤 Собеседование началось. Говорите… (скажите ‘стоп’ чтобы завершить)")
    print(
        f"🎚️ SR={samplerate}Hz, окно={chunk_sec:.1f}s, пауза={pause_chunks} окна, RMS_GATE={rms_gate}"
    )

    running = True

    def _sigint(_sig, _frm):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sigint)

    buffer = []
    silence_count = 0
    sd.default.samplerate = samplerate
    if args.device_index is not None:
        sd.default.device = (args.device_index, None)

    while running:
        # 1) запишем 1 сек
        audio = sd.rec(
            int(samplerate * chunk_sec),
            samplerate=samplerate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        audio = audio.flatten()

        # 2) VAD по секундному окну
        x = torch.tensor(audio, dtype=torch.float32, device=_VAD_DEVICE)
        try:
            ts = get_speech_timestamps(x, vad_model, sampling_rate=samplerate)
            has_speech_silero = len(ts) > 0
        except Exception:
            has_speech_silero = False

        rms = float(np.sqrt(np.mean(audio * audio)))
        has_speech_rms = rms > rms_gate
        has_speech = has_speech_silero or has_speech_rms
        print(
            f"ℹ️ VAD 1s: rms={rms:.4f} has_speech={has_speech} (silero={has_speech_silero}, rms_gate={has_speech_rms})"
        )

        if has_speech:
            buffer.append(audio)
            silence_count = 0
        else:
            silence_count += 1

        # 3) если была речь и теперь пауза X*1s → ASR
        if silence_count >= pause_chunks and buffer:
            chunk = np.concatenate(buffer)
            buffer.clear()
            silence_count = 0
            # нормализация
            m = float(np.max(np.abs(chunk))) if chunk.size else 0.0
            if m > 0:
                chunk = (chunk / m).astype(np.float32)
            dur = chunk.size / samplerate
            rms_seg = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
            print(f"🎙️ сегмент {dur:.2f}s, rms={rms_seg:.4f} → ASR")
            if chunk.size < samplerate:  # менее 1 секунды — скипаем
                continue
            try:
                result = stt_model.transcribe(
                    chunk, language="ru", fp16=(device == "cuda")
                )
                user_text = (result.get("text") or "").strip()
            except Exception as e:
                print(f"⚠️ Whisper error: {e}")
                user_text = ""
            print(f"👤 Пользователь: {user_text}")
            if any(sw in user_text.lower() for sw in STOP_WORDS):
                break

        time.sleep(0.01)

    print("⏹ Остановка")


if __name__ == "__main__":
    main()
