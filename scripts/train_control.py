"""Start, pause, resume, and inspect continued Mini-K3 pretraining."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def process_alive(pid):
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid(cfg):
    path = ROOT / cfg["pid_file"]
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def read_json_lines(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, TypeError):
            pass
    return rows


def checkpoint_status(path, log_path):
    """Read tiny metadata only; status must never import torch or load model weights."""
    sidecar = path.with_suffix(path.suffix + ".status.json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    for row in reversed(read_json_lines(log_path)):
        if row.get("event") == "checkpoint":
            return {"updates": row.get("updates"), "epoch": row.get("epoch"),
                    "micro_step": row.get("micro_step"), "save_reason": row.get("reason")}
    return None


def command(cfg, resume_path):
    return [
        sys.executable, "-u", "trainer/train_minik3.py",
        "--config", cfg["config"], "--data", cfg["data"],
        "--tokenizer", cfg["tokenizer"], "--output", cfg["output"],
        "--resume", str(resume_path), "--pause-file", cfg["pause_file"],
        "--seq-len", str(cfg["seq_len"]), "--batch-size", str(cfg["batch_size"]),
        "--grad-accum", str(cfg["grad_accum"]), "--epochs", str(cfg["epochs"]),
        "--learning-rate", str(cfg["learning_rate"]), "--warmup-steps", str(cfg["warmup_steps"]),
        "--max-steps", str(cfg["max_steps"]), "--save-every", str(cfg["save_every"]),
        "--log-every", str(cfg["log_every"]), "--device", cfg["device"],
    ]


def start(cfg):
    pid = read_pid(cfg)
    if pid and process_alive(pid):
        print(f"already running: PID {pid}")
        return
    pause = ROOT / cfg["pause_file"]
    pause.unlink(missing_ok=True)
    output = ROOT / cfg["output"]
    initial = ROOT / cfg["initial_checkpoint"]
    resume_path = output if output.exists() else initial
    if not resume_path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
    log, err = ROOT / cfg["log"], ROOT / cfg["error_log"]
    log.parent.mkdir(parents=True, exist_ok=True)
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    with log.open("a", encoding="utf-8") as stdout, err.open("a", encoding="utf-8") as stderr:
        proc = subprocess.Popen(command(cfg, resume_path), cwd=ROOT, stdout=stdout, stderr=stderr,
                                creationflags=flags, start_new_session=(os.name != "nt"))
    (ROOT / cfg["pid_file"]).write_text(str(proc.pid), encoding="ascii")
    print(f"started PID {proc.pid}; resume={resume_path.relative_to(ROOT)}")


def pause(cfg):
    pid = read_pid(cfg)
    if not pid or not process_alive(pid):
        print("not running")
        return
    path = ROOT / cfg["pause_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    print(f"pause requested for PID {pid}; it will save after the current optimizer step")


def status(cfg):
    pid = read_pid(cfg)
    alive = bool(pid and process_alive(pid))
    output = ROOT / cfg["output"]
    log = ROOT / cfg["log"]
    saved = checkpoint_status(output, log)
    rows = read_json_lines(log)
    latest = next((row for row in reversed(rows) if "update" in row), None)
    print(json.dumps({"running": alive, "pid": pid, "checkpoint": str(output.relative_to(ROOT)),
                      "checkpoint_updates": saved.get("updates") if saved else None,
                      "latest_logged_update": latest.get("update") if latest else None,
                      "target_updates": cfg["max_steps"]}, indent=2))
    if rows:
        print("last log:", json.dumps(rows[-1], ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("start", "pause", "resume", "status"))
    p.add_argument("--config", default="configs/continued_pretrain.json")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.action in ("start", "resume"):
        start(cfg)
    elif args.action == "pause":
        pause(cfg)
    else:
        status(cfg)


if __name__ == "__main__":
    main()
