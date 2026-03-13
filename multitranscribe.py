#!/usr/bin/env python3
from faster_whisper import WhisperModel
import os

INPUT_DIR  = r"C:\Users\lucie\OneDrive\Dokumente\Studium\MasterInformatik\Cultural Analytics\NS Film Audio"
OUTPUT_DIR = os.path.join(INPUT_DIR, "Transkripte")
AUDIO_EXTS = {".mp3", ".mp4", ".wav", ".m4a", ".flac"}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Lade Modell...")
    model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=10, num_workers=2)
    print("✔ Modell bereit.\n")

    audio_files = [
        os.path.join(INPUT_DIR, f)
        for f in sorted(os.listdir(INPUT_DIR))
        if os.path.splitext(f)[1].lower() in AUDIO_EXTS
    ]

    todo = [
        f for f in audio_files
        if not os.path.exists(
            os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(f))[0] + ".txt")
        )
    ]

    print(f"📂 Gesamt: {len(audio_files)} | ✔ Vorhanden: {len(audio_files)-len(todo)} | ▶ Todo: {len(todo)}\n")

    for i, audio_path in enumerate(todo, 1):
        basename    = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{basename}.txt")
        print(f"[{i}/{len(todo)}] {basename} ...")

        try:
            segments, info = model.transcribe(
                audio_path,
                language="de",
                beam_size=1,
                vad_filter=True
            )

            lines = [
                f"[{s.start/60:.0f}:{s.start%60:04.1f} → {s.end/60:.0f}:{s.end%60:04.1f}] {s.text.strip()}"
                for s in segments
            ]

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            print(f"    ✔ Fertig → {os.path.basename(output_path)}")

        except Exception as e:
            print(f"    ⚠ Fehler: {e}")

    print("\n✅ Alle Dateien verarbeitet.")

if __name__ == "__main__":
    main()
