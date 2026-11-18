"""Resilient local orchestrator for 5-fold OOF on Colab L4 with per-fold download & kernel-busy keepalive."""
from __future__ import annotations

import argparse
import hashlib
import json
import select
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path("/Users/sunggyu/Desktop/이성규/DCC")
SRC = ROOT / "artifacts_private/e026_same_input_replay_20261116/source"
CLI = "/tmp/dcc-colab-cli/bin/colab"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_cli(args: list[str], stdin_text: str | None = None, timeout: int = 90) -> subprocess.CompletedProcess:
    return subprocess.run([CLI] + args, input=stdin_text, capture_output=True, text=True, timeout=timeout)


def setup_session(sess: str) -> None:
    check_sess = run_cli(["sessions"], timeout=30)
    if sess in check_sess.stdout:
        print(f"[orchestrator] session {sess} already active; checking /content/m4_precue...", flush=True)
        chk = run_cli(["ls", "-s", sess, "/content/m4_precue/inputs"], timeout=30)
        if chk.returncode == 0 and "train.jsonl" in chk.stdout:
            run_cli(["upload", "-s", sess, str(ROOT / "tools/m4_precue_bce/remote_run_oof_fold.py"), "/content/remote_run_oof_fold.py"], timeout=60).check_returncode()
            print("[orchestrator] reused active session & updated remote_run_oof_fold.py.", flush=True)
            return
    else:
        print(f"[orchestrator] creating session {sess}...", flush=True)
        r = run_cli(["new", "--session", sess, "--gpu", "L4"], timeout=90)
        if r.returncode != 0:
            raise RuntimeError(f"Session creation failed: {r.stdout}\n{r.stderr}")

    run_cli(["upload", "-s", sess, str(ROOT / "tools/full22k/chunk_checkpoint.py"), "/content/chunk_checkpoint.py"], timeout=60).check_returncode()
    rdir = run_cli(["exec", "-s", sess, "--timeout", "30"], stdin_text="import os; os.makedirs('/content/chunks', exist_ok=True); print('DIR_READY')", timeout=45)
    if rdir.returncode != 0:
        raise RuntimeError(f"Failed creating /content/chunks: {rdir.stdout}\n{rdir.stderr}")

    chunk_files = sorted(Path("/tmp/m4_precue_chunks").iterdir())
    def up(p: Path):
        for att in range(4):
            try:
                res = run_cli(["upload", "-s", sess, str(p), f"/content/chunks/{p.name}"], timeout=120)
                if res.returncode == 0:
                    return p.name
            except subprocess.TimeoutExpired:
                pass
            time.sleep(2 * (att + 1))
        raise RuntimeError(f"Upload failed: {p.name}")

    done = 0
    with ThreadPoolExecutor(max_workers=2) as ex:
        for fut in as_completed([ex.submit(up, p) for p in chunk_files]):
            fut.result()
            done += 1
            if done % 9 == 0 or done == len(chunk_files):
                print(f"[orchestrator] uploaded {done}/{len(chunk_files)} chunks", flush=True)

    join_code = (ROOT / "tools/m4_precue_bce/join_and_setup.py").read_text(encoding="utf-8")
    rjoin = run_cli(["exec", "-s", sess, "--timeout", "90"], stdin_text=join_code, timeout=120)
    if rjoin.returncode != 0:
        raise RuntimeError(f"Extract failed: {rjoin.stdout}\n{rjoin.stderr}")

    run_cli(["upload", "-s", sess, str(ROOT / "tools/m4_precue_bce/train_precue.py"), "/content/m4_precue/train_precue.py"], timeout=60).check_returncode()
    run_cli(["upload", "-s", sess, str(ROOT / "tools/m4_precue_bce/remote_run_oof_fold.py"), "/content/remote_run_oof_fold.py"], timeout=60).check_returncode()
    print("[orchestrator] session setup complete.", flush=True)


def run_fold_foreground(sess: str, oof_name: str, fold: int, seed: int, precue_pw: float, jw: float, mfw: float) -> None:
    cmd = (
        "import subprocess, sys\n"
        "p = subprocess.Popen([\n"
        "    sys.executable, '-u', '/content/remote_run_oof_fold.py',\n"
        f"    '--oof-name', '{oof_name}', '--fold', '{fold}', '--seed', '{seed}',\n"
        f"    '--precue-pos-weight', '{precue_pw}', '--jaccard-weight', '{jw}', '--macro-f1-weight', '{mfw}',\n"
        "], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
        "for line in p.stdout:\n"
        "    print(line, end='', flush=True)\n"
        "raise SystemExit(p.wait())\n"
    )
    for attempt in range(6):
        print(f"[orchestrator] fold {fold} foreground exec attempt {attempt + 1}...", flush=True)
        try:
            proc = subprocess.Popen(
                [CLI, "exec", "-s", sess, "--timeout", "300"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            assert proc.stdin and proc.stdout
            proc.stdin.write(cmd)
            proc.stdin.close()
            t_last = time.time()
            while True:
                ready, _, _ = select.select([proc.stdout], [], [], 5.0)
                if ready:
                    line = proc.stdout.readline()
                    if not line:
                        if proc.poll() is not None:
                            break
                    else:
                        t_last = time.time()
                        print(line, end="", flush=True)
                        continue
                if proc.poll() is not None:
                    for rem in proc.stdout:
                        print(rem, end="", flush=True)
                    break
                if time.time() - t_last > 360:
                    proc.kill()
                    proc.wait(timeout=10)
                    raise TimeoutError("No output line from colab exec for 360s")
            if proc.returncode == 0:
                return
            print(f"[orchestrator] fold {fold} exec exited rc={proc.returncode}", flush=True)
        except Exception as exc:
            print(f"[orchestrator] fold {fold} transient error: {exc}", flush=True)
        ls_res = run_cli(["ls", "-s", sess, "/content"], timeout=30)
        if ls_res.returncode != 0:
            raise RuntimeError(f"Colab session {sess} is gone! Details: {ls_res.stdout} {ls_res.stderr}")
        try:
            run_cli(["restart-kernel", "-s", sess], timeout=45)
        except subprocess.TimeoutExpired:
            print(f"[orchestrator] restart-kernel timed out, retrying attach...", flush=True)
        time.sleep(3)
    raise RuntimeError(f"Fold {fold} failed after 6 retries")


def download_with_retry(sess: str, remote_path: str, local_path: Path, timeout: int = 90) -> None:
    last_err = ""
    for att in range(4):
        try:
            r = run_cli(["download", "-s", sess, remote_path, str(local_path)], timeout=timeout)
            if r.returncode == 0:
                return
            last_err = f"rc={r.returncode} {r.stdout} {r.stderr}"
        except subprocess.TimeoutExpired:
            last_err = f"timeout({timeout}s)"
        time.sleep(2 * (att + 1))
    raise RuntimeError(f"Failed downloading {remote_path}: {last_err}")


def download_fold_outputs(sess: str, oof_name: str, fold: int, local_root: Path) -> None:
    fdir = local_root / f"fold{fold}"
    fdir.mkdir(parents=True, exist_ok=True)
    remote_dir = f"/content/oof_runs/{oof_name}/fold{fold}"
    for fname in ["oof_prefix.npz", "oof_prefix.meta.json", "run_config.json", "optimizer_audit.json", "initialization.json", "train_log.txt", "DONE.json"]:
        download_with_retry(sess, f"{remote_dir}/{fname}", fdir / fname, timeout=90)
    for split_fn in [f"fold{fold}_fit.txt", f"fold{fold}_held.txt"]:
        download_with_retry(sess, f"/content/oof_runs/{oof_name}/{split_fn}", local_root / split_fn, timeout=30)

    meta = json.loads((fdir / "oof_prefix.meta.json").read_text(encoding="utf-8"))
    done = json.loads((fdir / "DONE.json").read_text(encoding="utf-8"))
    sha = file_sha256(fdir / "oof_prefix.npz")
    assert sha == meta["npz_sha256"] == done["npz_sha256"], (sha, meta["npz_sha256"], done["npz_sha256"])
    print(f"[orchestrator] fold {fold} verified locally! npz_sha256={sha}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--oof-name", required=True)
    ap.add_argument("--local-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--precue-pos-weight", type=float, default=-1.0)
    ap.add_argument("--jaccard-weight", type=float, default=0.0)
    ap.add_argument("--macro-f1-weight", type=float, default=0.0)
    args = ap.parse_args()

    t0 = time.time()
    args.local_dir.mkdir(parents=True, exist_ok=True)
    setup_session(args.session)

    try:
        for k in range(5):
            local_done = args.local_dir / f"fold{k}/DONE.json"
            local_npz = args.local_dir / f"fold{k}/oof_prefix.npz"
            if local_done.exists() and local_npz.exists():
                print(f"[orchestrator] fold {k} already verified locally, skipping.", flush=True)
                continue
            run_fold_foreground(args.session, args.oof_name, k, args.seed, args.precue_pos_weight, args.jaccard_weight, args.macro_f1_weight)
            download_fold_outputs(args.session, args.oof_name, k, args.local_dir)
    finally:
        # Cleanly stop remote VM only if all 5 folds succeeded; if interrupted keep nothing billing? Wait: always stop if done!
        all_ok = all((args.local_dir / f"fold{k}/DONE.json").exists() for k in range(5))
        if all_ok:
            print(f"[orchestrator] All 5 folds done! Stopping session {args.session}...", flush=True)
            run_cli(["stop", "-s", args.session], timeout=60)

    sys.path.insert(0, str(SRC))
    from m4 import simulate as sim  # noqa: E402
    inp = Path("/Volumes/로컬 디스크/DCC/private_experiments/e026_same_input_replay_20261116/inputs")
    fit_ids = [line.strip() for line in (inp / "fit_ids.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    dev_ids = {line.strip() for line in (inp / "dev_ids.txt").read_text(encoding="utf-8").splitlines() if line.strip()}
    tables = [sim.PrefixTable.load(args.local_dir / f"fold{k}/oof_prefix.npz") for k in range(5)]
    merged_ids = [cid for t in tables for cid in t.call_ids]
    assert len(merged_ids) == len(fit_ids) and set(merged_ids) == set(fit_ids) and not (set(merged_ids) & dev_ids)
    merged_offsets = np.concatenate([[0], np.cumsum(np.concatenate([t.n_utt for t in tables]))]).astype(np.int64)
    merged_probs = np.concatenate([t.probs for t in tables], axis=0).astype(np.float32)
    merged_labels = np.concatenate([t.labels for t in tables], axis=0).astype(np.uint8)
    merged_path = args.local_dir / "fit_oof_merged.npz"
    sim.PrefixTable(merged_ids, merged_offsets, merged_probs, merged_labels).save(merged_path)
    summary = {
        "done": True,
        "oof_name": args.oof_name,
        "n_calls": len(merged_ids),
        "n_moments": int(merged_probs.shape[0]),
        "merged_sha256": file_sha256(merged_path),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (args.local_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("ORCHESTRATOR_COMPLETE:", json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
