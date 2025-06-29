import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import sys
import subprocess
import whisper
import tempfile
import math
import time
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"# Frag sec

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.abspath(relative_path)


FFMPEG_PATH = get_resource_path("")#your ffmpeg binary path
FFPROBE_PATH = get_resource_path("")#your ffprobe binary path

ffmpeg_dir = os.path.dirname(get_resource_path(""))#ffmpeg and ffprobe binary folder path
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

def log(msg, tag=None):
    log_output.insert(tk.END, msg + "\n", tag)
    log_output.see(tk.END)
    root.update()

def get_audio_duration(audio_path):
    try:
        result = subprocess.run(
            [FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        return float(result.stdout.strip())
    except Exception as e:
        log(f"Error getting duration: {e}")
        return 0

def split_audio_ffmpeg(input_path, chunk_length, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    command = [
        FFMPEG_PATH, "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-f", "segment",
        "-segment_time", str(chunk_length),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        os.path.join(output_dir, "chunk_%03d.wav")
    ]
    subprocess.run(command, check=True)

def assign_speakers(chunk_files):
    encoder = VoiceEncoder()
    embeddings = []

    for chunk_path in chunk_files:
        wav = preprocess_wav(str(chunk_path))
        embed = encoder.embed_utterance(wav)
        embeddings.append(embed)

    embeddings = np.array(embeddings)

    n_clusters = min(len(embeddings), 4)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
    labels = clustering.fit_predict(embeddings)

    return labels

def transcribe_with_speakers(audio_path, output_path, model_name, language, device):
    try:
        log(" Loading model...")
        model_path = get_resource_path(os.path.join("models", f"{model_name}.pt"))#modify to your model path, .pt files.
        model = whisper.load_model(model_path, device=device)
        log(" Model loaded.")

        duration = get_audio_duration(audio_path)
        if duration == 0:
            log(" Could not determine audio duration.")
            return

        chunk_duration = 15
        num_chunks = math.ceil(duration / chunk_duration)
        file_size = os.path.getsize(audio_path) / (1024 * 1024)

        with tempfile.TemporaryDirectory() as tempdir:
            log(" Splitting audio into chunks...")
            split_audio_ffmpeg(audio_path, chunk_duration, tempdir)
            chunk_files = sorted(Path(tempdir).glob("chunk_*.wav"))

            log(" Analyzing speaker voices...")
            speaker_labels = assign_speakers(chunk_files)
            log(f" Detected {len(set(speaker_labels))} speakers.")

            start_time = time.time()

            with open(output_path, "w", encoding="utf-8") as out_file:
                for i, chunk_path in enumerate(chunk_files):
                    result = model.transcribe(str(chunk_path), language=language)
                    text = result["text"].strip()
                    speaker = f"Speaker {speaker_labels[i]+1}"
                    out_file.write(f"{speaker}: {text}\n")
                    log(f"{speaker}: {text}", tag="transcript")

                    if device == "cuda":
                        torch.cuda.empty_cache()

                    percent = int((i + 1) / num_chunks * 100)
                    progress_var.set(percent)
                    root.update()

            elapsed = time.time() - start_time
            speed = file_size / elapsed
            log(f"\n Done in {elapsed:.2f}s — Avg speed: {speed:.2f} MB/s")
            messagebox.showinfo("Done", f"Transcription saved to:\n{output_path}")

    except Exception as e:
        log(f" Error: {e}")
        messagebox.showerror("Error", str(e))

def start_transcription():
    audio_path = entry_audio.get()
    output_path = entry_output.get()
    model_name = model_var.get()
    language = language_var.get()
    device = device_var.get()

    if not os.path.isfile(audio_path):
        messagebox.showerror("Error", "Invalid audio file path.")
        return
    if not output_path.endswith(".txt"):
        messagebox.showerror("Error", "Output must end with .txt")
        return

    log_output.delete("1.0", tk.END)
    progress_var.set(0)
    threading.Thread(target=transcribe_with_speakers,
                     args=(audio_path, output_path, model_name, language, device)).start()

def browse_audio():
    path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg *.webm *.aac")])
    if path:
        entry_audio.delete(0, tk.END)
        entry_audio.insert(0, path)

def browse_output():
    path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
    if path:
        entry_output.delete(0, tk.END)
        entry_output.insert(0, path)

root = tk.Tk()
root.title("transcriber V4")

for i in range(3):
    root.grid_columnconfigure(i, weight=1)
for r in range(8):
    root.grid_rowconfigure(r, weight=1)

tk.Label(root, text="Audio file:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
entry_audio = tk.Entry(root)
entry_audio.grid(row=0, column=1, sticky="we", padx=5)
tk.Button(root, text="Browse", command=browse_audio).grid(row=0, column=2, sticky="we", padx=5)

tk.Label(root, text="Output file (.txt):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
entry_output = tk.Entry(root)
entry_output.grid(row=1, column=1, sticky="we", padx=5)
tk.Button(root, text="Browse", command=browse_output).grid(row=1, column=2, sticky="we", padx=5)

tk.Label(root, text="Model:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
model_var = tk.StringVar(value="small")
tk.OptionMenu(root, model_var, "tiny", "base", "small", "medium", "large-v3").grid(row=2, column=1, sticky="w", padx=5)

tk.Label(root, text="Language code:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
language_var = tk.StringVar(value="en")
tk.Entry(root, textvariable=language_var, width=10).grid(row=3, column=1, sticky="w", padx=5)

tk.Label(root, text="Device:").grid(row=4, column=0, sticky="w", padx=5, pady=2)
device_var = tk.StringVar(value="cpu")
tk.OptionMenu(root, device_var, "cpu", "cuda").grid(row=4, column=1, sticky="w", padx=5)

tk.Button(root, text="Start Transcription", command=start_transcription).grid(row=5, column=1, sticky="we", pady=10)

progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
progress_bar.grid(row=6, column=0, columnspan=3, sticky="we", padx=5, pady=5)

log_output = tk.Text(root, height=12, wrap=tk.WORD, bg="black", fg="lime", insertbackground="white")
log_output.grid(row=7, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
log_output.tag_config("transcript", foreground="white")

root.mainloop()
