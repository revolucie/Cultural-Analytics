#!/usr/bin/env python3
import os
import subprocess
import sys

INPUT_DIR  = r"C:\Users\lucie\OneDrive\Dokumente\Studium\MasterInformatik\Cultural Analytics\NS-Filme"
OUTPUT_DIR = r"C:\Users\lucie\OneDrive\Dokumente\Studium\MasterInformatik\Cultural Analytics\NS Film Audio"

def convert_mp4_to_mp3(input_path, output_path):
    cmd = [
        "ffmpeg",
        "-i", input_path,       # Eingabedatei
        "-vn",                  # kein Video
        "-ar", "16000",         # 16kHz Samplerate (optimal für Whisper)
        "-ac", "1",             # Mono (Whisper braucht kein Stereo)
        "-q:a", "2",            # gute Qualität
        "-y",                   # überschreiben ohne Nachfrage
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mp4_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".mp4")
    ])

    if not mp4_files:
        print("Keine MP4-Dateien gefunden.", file=sys.stderr)
        sys.exit(1)

    print(f"Gefundene MP4-Dateien: {len(mp4_files)}\n")

    # Debug: zeige alle bereits vorhandenen MP3s
    existing_mp3s = {f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".mp3")}
    print(f"Bereits vorhandene MP3s: {len(existing_mp3s)}")
    for f in existing_mp3s:
        print(f"  {f}")


    ok_count  = 0
    err_count = 0

    for i, filename in enumerate(mp4_files, 1):
        basename    = os.path.splitext(filename)[0]
        input_path  = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f"{basename}.mp3")

        if os.path.exists(output_path):
            print(f"[{i}/{len(mp4_files)}] Überspringe (existiert bereits): {basename}.mp3")
            ok_count += 1
            continue

        print(f"[{i}/{len(mp4_files)}] Konvertiere: {filename} → {basename}.mp3 ...")
        success, stderr = convert_mp4_to_mp3(input_path, output_path)

        if success:
            print(f"  ✔ Fertig")
            ok_count += 1
        else:
            print(f"  ⚠ Fehler bei {filename}:")
            print(f"  {stderr[-300:]}")  # letzte 300 Zeichen der Fehlermeldung
            err_count += 1

    print(f"\n✅ Fertig: {ok_count} erfolgreich, {err_count} Fehler")

if __name__ == "__main__":
    main()
