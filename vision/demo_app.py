"""(vision/demo_app.py) Room check demo in the browser: v3 counting (single) vs hybrid, plus new items.

    python demo_app.py --adapter /tmp/vl_cmp/v3 [--examples DIR] [--port 7860] [--no-tunnel]
    (on Kaggle: bash run_kaggle.sh demo -> prints "OPEN THIS: https://....trycloudflare.com")

One Qwen3-VL-4B in memory; the v3 LoRA is switched on for counting / boxes and off for "new items" (the base model
lists other objects better: 80% vs 26% in the bench).
  single  the v3 counting answer
  hybrid  the v3 counting answer decides 0 / not 0; if not 0, "locate every X" and count the boxes
Expected items: type them ("pillow 4, towel 2, lamp 2") or leave the counts out and give a BEFORE photo: the before
counts are then read from that photo. New items: with a before photo, only objects listed after and not before.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request

from common import clean_new_items, same_object

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#9a6324", "#469990", "#000075"]


def start_tunnel(port: int) -> subprocess.Popen:
    """Cloudflare quick tunnel -> public https URL (printed), as in the voice demo. Downloads cloudflared if needed."""
    exe = shutil.which("cloudflared") or "/tmp/cloudflared"
    if not os.path.exists(exe):
        print("downloading cloudflared ...", flush=True)
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", exe)
        os.chmod(exe, 0o755)
    proc = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", f"http://localhost:{port}"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def watch():
        for line in proc.stdout:
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if m:
                print(f"\nOPEN THIS: {m.group(0)}\n", flush=True)

    threading.Thread(target=watch, daemon=True).start()
    return proc


def parse_expected(text: str) -> tuple[list[str], dict]:
    """'pillow 4, towel: 2, lamp' -> (names, {name: count}) ; a name without a number has no before count."""
    names, counts = [], {}
    for part in re.split(r"[,\n;]+", text or ""):
        part = part.strip().lower()
        if not part:
            continue
        m = re.match(r"^(.*?)[\s:x=]*(\d+)$", part)
        name, n = (m.group(1).strip(), int(m.group(2))) if m else (part, None)
        if name and name not in names:
            names.append(name)
            if n is not None:
                counts[name] = n
    return names, counts


def draw_boxes(path: str, boxes: dict):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(path).convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img)
    lw = max(2, round(min(w, h) / 250))
    try:
        font = ImageFont.load_default(size=max(14, min(w, h) // 35))
    except TypeError:                                   # Pillow < 10.1
        font = ImageFont.load_default()
    for k, (name, bs) in enumerate(boxes.items()):
        col = COLORS[k % len(COLORS)]
        for i, (x1, y1, x2, y2) in enumerate(bs or []):
            xy = [x1 * w / 1000, y1 * h / 1000, x2 * w / 1000, y2 * h / 1000]   # Qwen boxes: 0-1000 relative
            d.rectangle(xy, outline=col, width=lw)
            tb = d.textbbox((xy[0] + lw, xy[1] + lw), f"{name} {i + 1}", font=font)
            d.rectangle(tb, fill=col)
            d.text((xy[0] + lw, xy[1] + lw), f"{name} {i + 1}", fill="white", font=font)
    return img


def status(before, after, tol: int) -> str:
    if after is None:
        return "?"
    if before is None:
        return ""
    if before > 0 and after == 0:
        return "MISSING"
    if abs(after - before) <= tol:
        return "ok"
    return "fewer" if after < before else "more"


def check(model, after_img, before_img, expected_text, mode, want_new, tolerance):
    """One room check; returns (boxed image or None, table rows, new-items markdown, info markdown)."""
    if True:  # (kept at this indentation: the body was the UI callback)
        t0 = time.time()
        if not after_img:
            return None, [], "", "Upload an AFTER photo."
        names, before = parse_expected(expected_text)
        if not names:
            return None, [], "", "Type the expected items, e.g. pillow 4, towel 2, lamp 2, chair 1"
        note = []
        if before_img and len(before) < len(names):                 # missing before counts: read the before photo
            got, _ = model.count([before_img], [names])[0]
            for n in names:
                before.setdefault(n, got.get(n))
            note.append("before counts read from the BEFORE photo")
        single, raw = model.count([after_img], [names])[0]
        boxes = {}
        hybrid = dict(single)
        if mode != "single":
            ask = [n for n in names if single.get(n)]                # only items the count says are there
            if ask:
                for n, (bs, _) in zip(ask, model.locate([(after_img, n) for n in ask])):
                    boxes[n] = bs or []
                    if bs is not None:
                        hybrid[n] = len(bs)
        rows = []
        for n in names:
            b = before.get(n)
            rows.append([n, "" if b is None else b, single.get(n, "?"), status(b, single.get(n), tolerance)]
                        + ([hybrid.get(n, "?"), status(b, hybrid.get(n), tolerance)] if mode != "single" else ["-", "-"]))
        new_txt = ""
        if want_new:
            after_new = clean_new_items(model.new_items([after_img], [names], base_only=True)[0][0], names)
            if before_img:
                before_new = clean_new_items(model.new_items([before_img], [names], base_only=True)[0][0], names)
                fresh = {k: v for k, v in after_new.items() if not any(same_object(k, b) for b in before_new)}
                more = {k: v for k, v in after_new.items()
                        if k in before_new and v > before_new[k]}
                new_txt = ("**New since the before photo:** " + (", ".join(f"{k} ({v})" for k, v in fresh.items())
                                                              or "none")
                           + ("\n\n**More than before:** " + ", ".join(f"{k} {before_new[k]} -> {v}"
                                                                       for k, v in more.items()) if more else ""))
            else:
                new_txt = ("**Other objects in the after photo** (no before photo to compare, so everything is "
                           "listed): " + (", ".join(f"{k} ({v})" for k, v in after_new.items()) or "none"))
        img = draw_boxes(after_img, boxes) if boxes else None
        info = f"{time.time() - t0:.1f} s" + (f" | {'; '.join(note)}" if note else "") + f"\n\nraw count answer: `{raw.strip()[:400]}`"
        return img, rows, new_txt, info


def build(model, examples: list[str]):
    import gradio as gr

    def run(*args):
        return check(model, *args)

    with gr.Blocks(title="Room check") as app:
        gr.Markdown("## Room check: v3 counting vs hybrid\nAfter photo + expected items -> what is missing, changed, new.")
        with gr.Row():
            with gr.Column():
                after = gr.Image(type="filepath", label="AFTER photo (required)")
                before = gr.Image(type="filepath", label="BEFORE photo (optional: before counts + new-item comparison)")
            with gr.Column():
                expected = gr.Textbox(label="Expected items (before inventory)", value="pillow 4, towel 2, lamp 2, chair 1",
                                      lines=3, info="item and count; leave the count out to read it from the BEFORE photo")
                mode = gr.Radio(["single", "hybrid", "both"], value="both", label="Counting",
                                info="single = v3 answer | hybrid = v3 decides 0 / not 0, boxes give how many")
                want_new = gr.Checkbox(value=True, label="Also list new items (base model, LoRA off)")
                tol = gr.Slider(0, 2, value=0, step=1, label="Tolerance (difference still 'ok')")
                go = gr.Button("Check room", variant="primary")
        table = gr.Dataframe(headers=["item", "before", "single", "single status", "hybrid", "hybrid status"],
                             label="Counts", wrap=True)
        with gr.Row():
            boxed = gr.Image(label="Hybrid boxes on the after photo")
            with gr.Column():
                new_out = gr.Markdown()
                info = gr.Markdown()
        go.click(run, [after, before, expected, mode, want_new, tol], [boxed, table, new_out, info])
        if examples:
            gr.Examples([[e, None, "pillow, towel, lamp, chair", "both", True, 0] for e in examples[:12]],
                        [after, before, expected, mode, want_new, tol], label="Example rooms (your frozen eval set)")
    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--adapter", required=True, help="the v3 LoRA folder (run_kaggle.sh demo downloads it)")
    p.add_argument("--examples", default="", help="folder of example photos")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--no-tunnel", action="store_true")
    a = p.parse_args()
    from vl_infer import RoomModel

    print("loading the model ...", flush=True)
    model = RoomModel(a.model, adapter=a.adapter)
    ex = sorted(glob.glob(os.path.join(a.examples, "*.jpg")))[:12] if a.examples else []
    app = build(model, ex)
    if not a.no_tunnel:
        start_tunnel(a.port)
    app.queue().launch(server_name="0.0.0.0", server_port=a.port, allowed_paths=[a.examples or tempfile.gettempdir()])


if __name__ == "__main__":
    main()
