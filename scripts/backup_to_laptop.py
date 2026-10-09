#!/usr/bin/env python3
"""Continuously mirror completed server artifacts to a SHA256-verified laptop backup.

Requires only Python and the OpenSSH client. Interrupted transfers resume at the
last verified file; incomplete files never replace an existing good backup.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time


GIB = 1024 ** 3
CRITICAL_MODELS = {"x2_mit_bw_r90", "x3_unetpp_bw_e120_r90"}
MODEL_SUFFIXES = {".pt", ".pth", ".pkl"}
RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)
}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe_relative(path):
    """Reject names that can escape, alias, or collide on a Windows filesystem."""
    if not isinstance(path, str) or not path or any(c in path for c in "\\:\x00"):
        raise ValueError(f"Unsafe inventory path: {path!r}")
    parts = path.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"Unsafe inventory path: {path!r}")
    for part in parts:
        if part.endswith((".", " ")) or part.split(".")[0].upper() in RESERVED:
            raise ValueError(f"Windows-incompatible inventory path: {path!r}")
        if any(ord(c) < 32 or c in '<>"|?*' for c in part):
            raise ValueError(f"Windows-incompatible inventory path: {path!r}")
    return Path(*parts)


def priority(path):
    parts = path.split("/")
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        return 2
    if (len(parts) == 5 and parts[:2] == ["runs", "cv"] and parts[2] in CRITICAL_MODELS
            and re.fullmatch(r"fold[0-4]", parts[3]) and parts[4] == "last.pt"):
        return 0
    if len(parts) > 1 and parts[:2] == ["runs", "si_clf"]:
        return 0
    if any(p in {"external_data", "external", "external_datasets"} for p in parts):
        return 1
    return 1 if suffix in MODEL_SUFFIXES else 0


def order(entry):
    path = entry["path"]
    p = priority(path)
    critical = p == 0 and Path(path).suffix.lower() == ".pt"
    if p == 0:
        rank = 0 if critical else (1 if path.startswith("data/") else 2)
    elif p == 1:
        rank = 0 if path.endswith("/last.pt") else (1 if Path(path).suffix.lower() in MODEL_SUFFIXES else 2)
    else:
        rank = 0
    return p, rank, path


def atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class Backup:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.backup_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.project = self.root / "server_project"
        self.project.mkdir(exist_ok=True)
        self.staging = self.root / "_partial"
        self.staging.mkdir(exist_ok=True)
        self.status = {"started_utc": utc_now(), "state": "starting", "phase": 0}
        self.inventory = []
        self.selected = []
        self.index = {}
        self.complete_paths = set()
        self.last_status = 0
        self.lock = None
        self.proc = None

    def log(self, message):
        print(f"{utc_now()} {message}", flush=True)

    def acquire_lock(self):
        self.lock = (self.root / "backup.lock").open("a+b")
        self.lock.seek(0, os.SEEK_END)
        if self.lock.tell() == 0:
            self.lock.write(b"\0")
            self.lock.flush()
        self.lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.lock.close()
            self.lock = None
            raise RuntimeError("Another backup process is already running.") from exc

    def load_index(self):
        index_path = self.root / "verified_index.json"
        if index_path.exists():
            with index_path.open(encoding="utf-8") as stream:
                self.index = json.load(stream)
        journal_path = self.root / "verified_files.jsonl"
        if journal_path.exists():
            with journal_path.open(encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    try:
                        record = json.loads(line)
                        safe_relative(record["path"])
                        self.index[record["path"]] = record
                    except (ValueError, KeyError, TypeError):
                        self.log(f"Ignoring incomplete journal record {number}")

    def target(self, relative):
        destination = self.project / safe_relative(relative)
        if os.path.commonpath([str(self.project.resolve()), str(destination.resolve())]) != str(self.project.resolve()):
            raise ValueError(f"Destination escapes backup directory: {relative!r}")
        return destination

    def verified(self, entry):
        record = self.index.get(entry["path"], {})
        if any(record.get(k) != entry[k] for k in ("size", "mtime_ns")):
            return False
        if not re.fullmatch(r"[a-f0-9]{64}", record.get("sha256", "")):
            return False
        target = self.target(entry["path"])
        return target.is_file() and target.stat().st_size == entry["size"]

    def write_status(self, force=False, **values):
        self.status.update(values)
        now = time.monotonic()
        if force or now - self.last_status >= 10:
            complete = [f for f in self.selected if f["path"] in self.complete_paths]
            critical = [f for f in self.inventory if priority(f["path"]) == 0]
            critical_models = {
                f"runs/cv/{model}/fold{fold}/last.pt"
                for model in CRITICAL_MODELS for fold in range(5)
            }
            restore_assets = {
                f["path"] for f in self.inventory
                if f["path"].split("/")[0] in {"src", "scripts", "experiments", "reports", "splits", "data", "backup_metadata"}
                and priority(f["path"]) == 0
            }
            restore_assets |= critical_models | {"runs/si_clf/sub6_pair.pkl"}
            data_count = sum(f["path"].startswith("data/") for f in self.inventory)
            self.status.update(
                updated_utc=utc_now(), total_files=len(self.selected),
                total_bytes=sum(f["size"] for f in self.selected),
                verified_files=len(complete), verified_bytes=sum(f["size"] for f in complete),
                critical_complete=all(f["path"] in self.complete_paths for f in critical),
                critical_restore_complete=(data_count >= 190 and restore_assets <= self.complete_paths
                                           and any(p.startswith("src/") for p in restore_assets)
                                           and "splits/folds5.json" in self.complete_paths),
            )
            atomic_json(self.root / "backup_status.json", self.status)
            self.last_status = now

    def ssh_command(self, action):
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
                self.args.server,
                f"cd ~/komap && .venv/bin/python scripts/backup_export.py {action}"]

    @staticmethod
    def process_options():
        return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

    def fetch_inventory(self):
        result = subprocess.run(self.ssh_command("list"), capture_output=True,
                                timeout=180, **self.process_options())
        if result.stderr:
            self.log(result.stderr.decode("utf-8", errors="replace").strip())
        if result.returncode:
            raise RuntimeError(f"Inventory SSH exited with {result.returncode}")
        manifest = json.loads(result.stdout)
        if manifest.get("version") != 1 or not isinstance(manifest.get("files"), list):
            raise ValueError("Unsupported inventory response")
        seen = set()
        for entry in manifest["files"]:
            relative = safe_relative(entry["path"])
            key = str(relative).casefold()
            if key in seen:
                raise ValueError(f"Duplicate Windows inventory path: {entry['path']!r}")
            seen.add(key)
            if not isinstance(entry["size"], int) or entry["size"] < 0:
                raise ValueError(f"Invalid size: {entry['path']}")
            if not isinstance(entry["mtime_ns"], int) or entry["mtime_ns"] < 0:
                raise ValueError(f"Invalid mtime: {entry['path']}")
        atomic_json(self.root / "inventory_latest.json", manifest)
        self.inventory = manifest["files"]
        self.selected = sorted(
            (f for f in self.inventory if not self.args.critical_only or priority(f["path"]) == 0),
            key=order,
        )
        self.complete_paths = {f["path"] for f in self.inventory if self.verified(f)}
        self.write_status(force=True, state="inventory_received", error=None, deferred_checkpoints=manifest.get("deferred_checkpoints", []))

    def journal(self, record):
        with (self.root / "verified_files.jsonl").open("a+b") as stream:
            stream.seek(0, os.SEEK_END)
            if stream.tell():
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b"\n":
                    stream.write(b"\n")
            stream.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        self.index[record["path"]] = record
        self.complete_paths.add(record["path"])

    def receive_batch(self, entries):
        expected = {entry["path"]: entry for entry in entries}
        received = set()
        request = json.dumps({"files": entries}, ensure_ascii=False).encode("utf-8")
        self.proc = subprocess.Popen(self.ssh_command("archive"), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     **self.process_options())
        proc = self.proc

        def drain_stderr():
            for line in iter(proc.stderr.readline, b""):
                self.log("server: " + line.decode("utf-8", errors="replace").rstrip())

        stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
        stderr_thread.start()
        try:
            proc.stdin.write(request)
            proc.stdin.close()
            with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
                for member in archive:
                    path = member.name
                    if not member.isfile() or path not in expected or path in received:
                        raise ValueError(f"Unexpected archive member: {path!r}")
                    entry = expected[path]
                    digest = member.pax_headers.get("komap.sha256", "")
                    if member.size != entry["size"] or not re.fullmatch(r"[a-f0-9]{64}", digest):
                        raise ValueError(f"Invalid archive metadata: {path!r}")
                    if member.pax_headers.get("komap.mtime_ns") != str(entry["mtime_ns"]):
                        raise ValueError(f"Archive mtime differs from inventory: {path!r}")
                    target = self.target(path)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    partial = self.staging / (hashlib.sha256(path.encode("utf-8")).hexdigest() + ".partial")
                    hasher, copied = hashlib.sha256(), 0
                    new_phase = priority(path)
                    self.write_status(force=new_phase != self.status.get("phase"), state="copying", phase=new_phase, latest=path, current_file_bytes=0)
                    source = archive.extractfile(member)
                    with source, partial.open("wb") as output:
                        while True:
                            chunk = source.read(4 * 1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                            hasher.update(chunk)
                            copied += len(chunk)
                            self.write_status(current_file_bytes=copied)
                        output.flush()
                        os.fsync(output.fileno())
                    if copied != entry["size"] or hasher.hexdigest() != digest:
                        raise ValueError(f"SHA256/size verification failed: {path!r}")
                    os.utime(partial, ns=(entry["mtime_ns"], entry["mtime_ns"]))
                    os.replace(partial, target)
                    self.journal(dict(entry, sha256=digest, verified_utc=utc_now()))
                    received.add(path)
                    self.write_status(current_file_bytes=copied)
            # tarfile stops at the end marker; drain remaining tar padding first.
            while proc.stdout.read(1024 * 1024):
                pass
            returncode = proc.wait(timeout=60)
            stderr_thread.join(timeout=5)
            if returncode:
                raise RuntimeError(f"Archive SSH exited with {returncode}")
            missing = set(expected) - received
            if missing:
                self.log(f"Server deferred {len(missing)} changed files; next inventory will retry")
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            stderr_thread.join(timeout=5)
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                if pipe and not pipe.closed:
                    pipe.close()
            self.proc = None
            atomic_json(self.root / "verified_index.json", self.index)

    def sync_once(self):
        self.fetch_inventory()
        pending = [f for f in self.selected if not self.verified(f)]
        remaining = sum(f["size"] for f in pending)
        free = shutil.disk_usage(self.root).free
        if free < remaining + 20 * GIB:
            raise RuntimeError(f"Insufficient backup space: need {remaining / GIB:.1f} GiB plus 20 GiB reserve; free {free / GIB:.1f} GiB")
        self.log(f"Inventory: {len(self.selected)} selected, {len(pending)} pending, {remaining / GIB:.2f} GiB pending")
        batch, batch_bytes, batch_phase = [], 0, None
        for entry in pending:
            phase = priority(entry["path"])
            if batch and (batch_bytes + entry["size"] > GIB or phase != batch_phase):
                self.receive_batch(batch)
                batch, batch_bytes = [], 0
            batch.append(entry)
            batch_bytes += entry["size"]
            batch_phase = phase
        if batch:
            self.receive_batch(batch)
        finished = all(self.verified(f) for f in self.selected)
        self.write_status(force=True, state="snapshot_complete" if finished else "retry_pending", current_file_bytes=0)
        self.log(f"Snapshot {'complete' if finished else 'partially complete'}; {self.status['verified_files']}/{len(self.selected)} verified")
        return finished

    def run(self):
        self.acquire_lock()
        try:
            self.load_index()
            while True:
                try:
                    # Keep the laptop awake during active I/O without changing its
                    # saved power settings. Waiting between polls does not inhibit sleep.
                    if os.name == "nt":
                        import ctypes
                        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
                    try:
                        complete = self.sync_once()
                    finally:
                        if os.name == "nt":
                            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
                    if self.args.once or self.args.critical_only:
                        return 0 if complete else 2
                    self.write_status(force=True, state="waiting", next_poll_seconds=self.args.poll_seconds)
                    time.sleep(self.args.poll_seconds)
                except KeyboardInterrupt:
                    self.write_status(force=True, state="stopped")
                    return 130
                except Exception as exc:
                    self.log(f"Backup error: {type(exc).__name__}: {exc}")
                    self.write_status(force=True, state="retrying", error=str(exc), next_poll_seconds=30)
                    if self.args.once or self.args.critical_only:
                        return 1
                    time.sleep(30)
        finally:
            if self.lock:
                self.lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", default=r"D:\KoMaP_Backup_20261009")
    parser.add_argument("--server", default="komap")
    parser.add_argument("--once", action="store_true", help="Transfer one inventory snapshot then exit")
    parser.add_argument("--critical-only", action="store_true", help="Transfer phase 0 only, then exit")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        parser.error("--poll-seconds must be at least 10")
    try:
        return Backup(args).run()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Backup could not start: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
