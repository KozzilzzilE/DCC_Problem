"""Launch remote_run_softjaccard.py detached on Colab L4 and verify GPU training started."""
import subprocess
import sys
import time
from pathlib import Path

log_path = Path("/content/master_sj_s42.log")
pid_path = Path("/content/master_sj_s42.pid")
proc = subprocess.Popen(
    [sys.executable, "-u", "/content/remote_run_softjaccard.py"],
    stdout=log_path.open("w", encoding="utf-8"),
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    start_new_session=True,
)
pid_path.write_text(str(proc.pid), encoding="utf-8")
time.sleep(8)
print(f"DETACHED_PID={proc.pid} alive={proc.poll() is None} rc={proc.poll()}", flush=True)
print("--- master_sj_s42.log ---", flush=True)
print(log_path.read_text(encoding="utf-8")[-1500:], flush=True)
