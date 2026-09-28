"""Shared argparse helpers for all entry points."""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import urllib.request

from tqdm import tqdm

from s2s.config import Config, load_config


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/default.yaml", help="YAML config file")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override config values, e.g. --set train_talker.max_steps=100")
    return p


def config_from_args(args) -> Config:
    return load_config(args.config, args.set)


def download(url: str, dest: str) -> str:
    """Download with a progress bar; skips if the file already exists."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[download] exists: {dest}")
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    print(f"[download] {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dest)) as bar:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    shutil.move(tmp, dest)
    return dest


def extract_tar(path: str, dest_dir: str, marker: str) -> None:
    """Extract once; `marker` is a path that exists after a successful extraction."""
    if os.path.exists(marker):
        print(f"[extract] already extracted: {marker}")
        return
    print(f"[extract] {path} -> {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(path) as tar:
        try:
            tar.extractall(dest_dir, filter="data")
        except TypeError:  # Python < 3.12 without the filter argument
            tar.extractall(dest_dir)
