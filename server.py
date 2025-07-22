#!/usr/bin/env python3
"""
TCP‑Dienst
1. empfängt 16‑bit‑PCM (mono, 48 kHz)
2. transkribiert Deutsch → Whisper‑Text
3. übersetzt Deutsch → Englisch (Ollama)
4. synthetisiert englische Sprache (Silero‑TTS v3_en)
5. sendet WAV‑Container zurück
"""

import io, socket, struct, time, wave
from typing import Union

import numpy as np
import whisper, ollama
from silero_tts.silero_tts import SileroTTS

# ─────────────────── Modell‑Laden ──────────────────────────────
print("● Loading Whisper model …")
wh_model = whisper.load_model("base")

print("● Loading Silero‑TTS (EN v3) …")
SILERO_SR = 48_000
silero = SileroTTS(
    model_id="v3_en", language="en", speaker="en_2",
    sample_rate=SILERO_SR, device="cpu"
)

# ─────────────────── TCP‑Server ────────────────────────────────
HOST, PORT = "0.0.0.0", 6000
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.bind((HOST, PORT)); srv.listen(1)
print(f"◆ Server listening on {HOST}:{PORT}")

# ─────────────────── Pipeline‑Funktion ─────────────────────────
def transcribe_translate_speak(pcm_bytes: bytes) -> bytes:
    """PCM → übersetztes WAV (Bytes)"""
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # 1· Whisper STT
    start = time.time()
    try:
        result = wh_model.transcribe(pcm, language="de")
        de_text = result["text"].strip()
    except Exception as exc:
        print("  Whisper‑Fehler:", exc)
        de_text = ""

    if not de_text:
        # kein Text → 0.5 s Stille zurück
        print("  Whisper   – leer, sende Stille")
        dur = 0.5
        empty = np.zeros(int(dur * SILERO_SR), dtype=np.float32)
        wav_bytes = (empty * 32767).astype(np.int16).tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SILERO_SR)
            w.writeframes(wav_bytes)
        return buf.getvalue()

    print(f"  Whisper   {time.time()-start:.2f}s  ▶  {de_text}")

    # 2· Ollama‑Übersetzung
    start = time.time()
    en_text = ollama.chat(
        model="phi4-mini",
        messages=[
            {"role": "system",
             "content": "Translate German to English. Output only the translation."},
            {"role": "user", "content": de_text}
        ]
    )["message"]["content"].strip()
    print(f"  Ollama    {time.time()-start:.2f}s  ▶  {en_text}")

    # 3· Silero‑TTS
    start = time.time()
    wav_buf = io.BytesIO()
    silero.tts(en_text, wav_buf)
    wav_data = wav_buf.getvalue()
    dur = (len(wav_data)-44) / (2*SILERO_SR)
    print(f"  Silero    {time.time()-start:.2f}s  ▶  {dur:.2f}s Audio")
    return wav_data

# ─────────────────── Haupt‑Loop ─────────────────────────────────
while True:
    conn, addr = srv.accept()
    print(f"● Connection from {addr}")
    try:
        # Länge lesen
        size_raw = conn.recv(4)
        if len(size_raw) != 4:
            print("  ✖ length prefix missing")
            conn.close(); continue
        (nbytes,) = struct.unpack("!I", size_raw)
        print(f"  ▶ expecting {nbytes} bytes ({nbytes/2/SILERO_SR:.2f}s)")

        # PCM lesen
        pcm = b""
        while len(pcm) < nbytes:
            chunk = conn.recv(nbytes - len(pcm))
            if not chunk: break
            pcm += chunk
        print(f"  ✔ received {len(pcm)} bytes")

        if len(pcm) < nbytes:
            print("  ✖ incomplete audio – abort")
            conn.sendall(struct.pack("!I", 0))  # leere Antwort
            conn.close(); continue

        # Verarbeitung
        wav_out = transcribe_translate_speak(pcm)

        # Antwort senden
        conn.sendall(struct.pack("!I", len(wav_out)))
        conn.sendall(wav_out)
        print("✓ Reply sent\n")

    except Exception as e:
        print("Error:", e)
    finally:
        conn.close()
