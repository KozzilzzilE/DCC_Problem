"""Launch remote_run.py detached on Colab L4 and confirm GPU training started."""
import subprocess
import sys
import time
from pathlib import Path

log_path = Path("/content/master_s42.log")
pid_path = Path("/content/master_s42.pid")
proc = subprocess.Popen(
    [sys.executable, "-u", "/content/remote_run.py"],
    stdout=log_path.open("w", encoding="utf-8"),
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    start_new_session=True,
)
pid_path.write_text(str(proc.pid), encoding="utf-8")
time.sleep(8)
print(f"DETACHED_PID={proc.pid} alive={proc.poll() is None} rc={proc.poll()}", flush=True)
print("--- master_s42.log ---", flush=True)
print(log_path.read_text(encoding="utf-8")[-2000:], flush=True)
print("--- nvidia-smi ---", flush=True)
print(subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"]).decode().strip(), flush=True)
