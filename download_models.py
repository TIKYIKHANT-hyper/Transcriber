import os
import whisper

os.environ["XDG_CACHE_HOME"] = os.path.abspath("")#your model download path

models = ["tiny", "base", "small", "medium", "large"]
for name in models:
    print(f"Downloading {name}.")
    model = whisper.load_model(name, device="cpu")#cpu mode for non-crash
    del model

print("Done!")
