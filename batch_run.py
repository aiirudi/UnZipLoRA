#!/usr/bin/env python3
"""Loop train + infer over a JSON list of {content_dir, content_name, style_name}.

Each entry's three fields are exported as env vars before invoking train.sh / infer.sh.
The shells were patched so their top-level `export content_dir=...` lines use
${content_dir:-default}, picking up these env vars when present.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "instance_data" / "train_prompts.json"
DEFAULT_INSTANCE_ROOT = Path("/home/xzh/xzh/UnZipLoRA/instance_data")
EVAL_SCRIPT = Path("/home/xzh/xzh/lora_evaluate/evaluate.py")
DEFAULT_METRICS_JSON = REPO_ROOT / "metrics.json"

# evaluate.py stdout patterns — keep in sync with evaluate.py:217-220
METRIC_PATTERNS = {
    "clipI_style": re.compile(r"style-only CLIP-I score is ([0-9.]+)"),
    "dino_content": re.compile(r"content-only DINO score is ([0-9.]+)"),
    "csd_content": re.compile(r"content-only CSD-content score is ([0-9.]+)"),
    "csd_style": re.compile(r"style-only CSD-style score is ([0-9.]+)"),
}


def parse_metrics(text: str) -> dict | None:
    out = {}
    for key, pat in METRIC_PATTERNS.items():
        m = pat.search(text)
        if not m:
            return None
        out[key] = float(m.group(1))
    return out


def update_metrics_json(json_path: Path, content_dir: str, record: dict) -> None:
    """Read-modify-write keyed by content_dir. Atomic via tmp+rename."""
    data = {}
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text() or "{}")
        except json.JSONDecodeError:
            backup = json_path.with_suffix(json_path.suffix + ".bak")
            json_path.rename(backup)
            print(f"[warn] {json_path} was malformed; backed up to {backup}", flush=True)
            data = {}
    data[content_dir] = record
    tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(json_path)


def run_step(name: str, script: Path, env: dict, log_path: Path | None) -> int:
    print(f"\n=== {name}: {script.name} (content_dir={env['content_dir']}) ===", flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab") as logf:
            logf.write(f"\n\n==== {name} @ {time.strftime('%F %T')} ====\n".encode())
            proc = subprocess.run(["bash", str(script)], env=env, cwd=REPO_ROOT,
                                  stdout=logf, stderr=subprocess.STDOUT)
        return proc.returncode
    return subprocess.run(["bash", str(script)], env=env, cwd=REPO_ROOT).returncode


def run_eval(content_dir: str, env: dict, log_path: Path | None) -> tuple[int, dict | None]:
    """Run evaluate.py --profile <content_dir>, capture stdout, return (rc, metrics)."""
    print(f"\n=== EVAL: evaluate.py --profile {content_dir} ===", flush=True)
    proc = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT), "--profile", content_dir],
        env=env, cwd=EVAL_SCRIPT.parent,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as logf:
            logf.write(f"\n\n==== EVAL @ {time.strftime('%F %T')} ====\n")
            logf.write(proc.stdout)
    # Echo just the summary block to console (not the model-loading noise)
    for line in proc.stdout.splitlines():
        if "score is" in line or "calculate indate profile" in line or line.startswith("-" * 10):
            print(line, flush=True)
    if proc.returncode != 0:
        return proc.returncode, None
    metrics = parse_metrics(proc.stdout)
    return proc.returncode, metrics


def run_one(entry: dict, args, instance_root: Path) -> bool:
    content_dir = entry["content_dir"]
    instance_dir = instance_root / content_dir
    if not instance_dir.is_dir():
        print(f"[skip] {content_dir}: instance dir missing at {instance_dir}", flush=True)
        return False

    env = os.environ.copy()
    env["content_dir"] = content_dir
    env["content_name"] = entry["content_name"]
    env["style_name"] = entry["style_name"]

    log_dir = Path(args.log_dir) if args.log_dir else None
    log_path = (log_dir / f"{content_dir}.log") if log_dir else None

    if not args.skip_train:
        rc = run_step("TRAIN", REPO_ROOT / "train.sh", env, log_path)
        if rc != 0:
            print(f"[fail] train rc={rc} for {content_dir}", flush=True)
            return False

    if not args.skip_infer:
        rc = run_step("INFER", REPO_ROOT / "infer.sh", env, log_path)
        if rc != 0:
            print(f"[fail] infer rc={rc} for {content_dir}", flush=True)
            return False

    if not args.skip_eval:
        rc, metrics = run_eval(content_dir, env, log_path)
        if rc != 0:
            print(f"[fail] eval rc={rc} for {content_dir}", flush=True)
            return False
        if metrics is None:
            print(f"[fail] could not parse metrics from evaluate.py stdout for {content_dir}", flush=True)
            return False
        record = {**metrics, "evaluated_at": time.strftime("%F %T")}
        update_metrics_json(Path(args.metrics_json), content_dir, record)
        print(f"[metrics] {content_dir}: {metrics}", flush=True)

    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help="Path to the prompt list JSON")
    p.add_argument("--instance-root", default=str(DEFAULT_INSTANCE_ROOT),
                   help="Base dir holding each content_dir's training images "
                        "(must match INSTANCE_DIR in train.sh)")
    p.add_argument("--only", nargs="*", default=None,
                   help="Run only these content_dir entries")
    p.add_argument("--skip", nargs="*", default=None,
                   help="Skip these content_dir entries")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-infer", action="store_true")
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip running evaluate.py and updating metrics JSON")
    p.add_argument("--metrics-json", default=str(DEFAULT_METRICS_JSON),
                   help="Path to the JSON file where evaluation metrics are accumulated")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Keep going if one entry fails")
    p.add_argument("--log-dir", default=None,
                   help="If set, tee per-entry stdout/stderr into this directory")
    args = p.parse_args()

    with open(args.config) as f:
        entries = json.load(f)

    if args.only:
        wanted = set(args.only)
        entries = [e for e in entries if e["content_dir"] in wanted]
    if args.skip:
        unwanted = set(args.skip)
        entries = [e for e in entries if e["content_dir"] not in unwanted]

    if not entries:
        print("No entries to run.", flush=True)
        return

    instance_root = Path(args.instance_root)
    ok, failed = 0, []
    for i, entry in enumerate(entries, 1):
        header = f"[{i}/{len(entries)}] {entry['content_dir']}"
        print(f"\n########## {header} ##########", flush=True)
        if run_one(entry, args, instance_root):
            ok += 1
        else:
            failed.append(entry["content_dir"])
            if not args.continue_on_error:
                print("[stop] failure — pass --continue-on-error to keep going.", flush=True)
                break

    print(f"\nDone. {ok}/{len(entries)} succeeded.")
    if failed:
        print("Failed/skipped:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
