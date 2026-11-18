"""Join uploaded chunks and extract base workspace on Colab L4."""
import subprocess
import sys
import zipfile
from pathlib import Path

subprocess.run(
    [sys.executable, "/content/chunk_checkpoint.py", "join", "/content/chunks", "/content/m4_precue_bundle.zip"],
    check=True,
)
dst = Path("/content/m4_precue")
dst.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile("/content/m4_precue_bundle.zip") as zf:
    zf.extractall(dst)
print("EXTRACTED_OK", sorted(p.name for p in dst.iterdir()), flush=True)
