"""(vision/demo_app.py) Room check demo in the browser: step 1 BEFORE photo -> every object detected (no list to type)
-> inventory, step 2 AFTER photo -> missing / changed / new. v3 counting (single) vs hybrid.

    python demo_app.py --adapter /tmp/vl_cmp/v3 [--examples DIR] [--detect-prompt FILE] [--port 7860] [--no-tunnel]
    (on Kaggle: bash run_kaggle.sh demo -> prints "OPEN THIS: https://....trycloudflare.com")

One Qwen3-VL-4B in memory; the LoRA is off for open detection (the base model lists objects better: 80% vs 26% in the
bench) and on for counting / boxes of the inventory items.
  step 1  "detect every object" (common.DETECT_PROMPT, or your own prompt with --detect-prompt) -> boxes per label
          -> inventory "pillow 4, towel 2, ..." (editable)
  step 2  single  the v3 counting answer for each inventory item
          hybrid  the v3 answer decides 0 / not 0; if not 0, "locate every X" and count the boxes
          new     the same open detection on the AFTER photo; labels matching nothing in the inventory
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

from common import same_object

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


CHUNK = 8          # items per counting question: v3 was trained on lists of up to 8 (MAX_PRESENT + MAX_ZEROS)


def count_photo(model, img: str, names: list[str], mode: str):
    """single = the v3 counting answer; hybrid = 0 if single says 0, else the number of "locate every X" boxes.
    A long inventory is asked in groups of CHUNK items, the list length v3 was trained on."""
    single, raw = {}, []
    groups = [names[i:i + CHUNK] for i in range(0, len(names), CHUNK)]
    for counts, r in model.count([img] * len(groups), groups):
        single.update(counts)
        raw.append(r.strip())
    hybrid, boxes = dict(single), {}
    if mode != "single":
        ask = [n for n in names if single.get(n)]                    # only items the count says are there
        for n, (bs, _) in zip(ask, model.locate([(img, n) for n in ask]) if ask else []):
            boxes[n] = bs or []
            if bs is not None:
                hybrid[n] = len(bs)
    return single, hybrid, boxes, " | ".join(raw)


def detect_photo(model, img: str, area: str, prompt: str | None) -> tuple[dict, str]:
    """Open detection, no item list (base model): {label: [boxes]}, the biggest groups first."""
    found, raw = model.detect([img], area=area, prompt=prompt)[0]
    return dict(sorted(found.items(), key=lambda kv: -len(kv[1]))), raw


def detect_before(model, prompt, before_img, area, mode):
    """Step 1: detect every object on the BEFORE photo -> the inventory (filled into the text box, editable)."""
    t0 = time.time()
    if not before_img:
        return None, [], "", "Upload a BEFORE photo."
    found, raw = detect_photo(model, before_img, area, prompt)
    if not found:
        return None, [], "", f"Nothing detected.\n\nraw answer: `{raw.strip()[:400]}`"
    names = list(found)
    rows = [[n, len(found[n])] for n in names]
    if mode != "single":                                           # cross-check: what v3 counts for the same items
        single, _, _, _ = count_photo(model, before_img, names, "single")
        rows = [r + [single.get(r[0], "?")] for r in rows]
    inventory = ", ".join(f"{n} {len(found[n])}" for n in names)
    info = (f"{time.time() - t0:.1f} s | {len(names)} kinds of objects, {sum(map(len, found.values()))} boxes. "
            "The inventory box is filled in: delete what you do not want checked, fix a wrong count, then step 2."
            f"\n\nraw detection answer: `{raw.strip()[:400]}`")
    return draw_boxes(before_img, found), rows, inventory, info


def check_after(model, prompt, after_img, inventory_text, area, mode, want_new, tolerance):
    """Step 2: count the inventory items on the AFTER photo (v3) -> ok / MISSING / fewer / more; new items = objects
    detected on the AFTER photo that match nothing in the inventory."""
    t0 = time.time()
    if not after_img:
        return None, [], "", "Upload an AFTER photo."
    names, before = parse_expected(inventory_text)
    if not names:
        return None, [], "", "No inventory yet: run step 1 on the BEFORE photo (or type it, e.g. pillow 4, towel 2)."
    single, hybrid, boxes, raw = count_photo(model, after_img, names, mode)
    rows = []
    for n in names:
        b = before.get(n)
        rows.append([n, "" if b is None else b, single.get(n, "?"), status(b, single.get(n), tolerance)]
                    + ([hybrid.get(n, "?"), status(b, hybrid.get(n), tolerance)] if mode != "single" else ["-", "-"]))
    new_txt = ""
    if want_new:
        found, _ = detect_photo(model, after_img, area, prompt)
        fresh = {k: v for k, v in found.items() if not any(same_object(k, n) for n in names)}
        new_txt = "**New since the BEFORE photo:** " + (", ".join(f"{k} ({len(v)})" for k, v in fresh.items()) or "none")
        boxes.update({f"NEW {k}": v for k, v in fresh.items()})
    info = f"{time.time() - t0:.1f} s\n\nraw count answer: `{raw[:400]}`"
    return draw_boxes(after_img, boxes) if boxes else None, rows, new_txt, info


def build(model, examples: list[str], prompt: str | None = None):
    import gradio as gr

    with gr.Blocks(title="Room check") as app:
        gr.Markdown("## Room check (v3)\n**Step 1** detect every object on the BEFORE photo -> inventory (no list to "
                    "type). **Step 2** check the AFTER photo -> missing / changed / new.")
        with gr.Row():
            area = gr.Textbox(label="Area being inspected (optional)", placeholder="bedroom, bathroom, kitchen ...")
            mode = gr.Radio(["single", "hybrid", "both"], value="both", label="Counting on the AFTER photo",
                            info="single = v3 answer | hybrid = v3 decides 0 / not 0, boxes give how many")
            want_new = gr.Checkbox(value=True, label="New items on the AFTER photo")
            tol = gr.Slider(0, 2, value=0, step=1, label="Tolerance (difference still 'ok')")
        items = gr.Textbox(label="Inventory (filled by step 1, editable)", lines=3,
                           placeholder="filled by step 1, e.g. pillow 4, towel 2, lamp 2")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 1: BEFORE")
                before = gr.Image(type="filepath", label="BEFORE photo")
                go1 = gr.Button("1. Detect BEFORE", variant="primary")
                before_boxed = gr.Image(label="Everything detected on the BEFORE photo")
                before_table = gr.Dataframe(headers=["item", "detected (boxes)", "v3 count"], label="BEFORE inventory")
                before_info = gr.Markdown()
            with gr.Column():
                gr.Markdown("### Step 2: AFTER")
                after = gr.Image(type="filepath", label="AFTER photo")
                go2 = gr.Button("2. Check AFTER", variant="primary")
                after_boxed = gr.Image(label="Boxes on the AFTER photo (NEW = not in the inventory)")
                after_table = gr.Dataframe(headers=["item", "before", "single", "single status", "hybrid",
                                                    "hybrid status"], label="AFTER vs inventory", wrap=True)
                after_md = gr.Markdown()
                after_info = gr.Markdown()
        go1.click(lambda *a: detect_before(model, prompt, *a), [before, area, mode],
                  [before_boxed, before_table, items, before_info])
        go2.click(lambda *a: check_after(model, prompt, *a), [after, items, area, mode, want_new, tol],
                  [after_boxed, after_table, after_md, after_info])
        if examples:
            gr.Examples([[e] for e in examples[:12]], [before], label="Example rooms (your frozen eval set): "
                        "click one as BEFORE, then upload an AFTER photo")
    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--adapter", required=True, help="the v3 LoRA folder (run_kaggle.sh demo downloads it)")
    p.add_argument("--examples", default="", help="folder of example photos")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--no-tunnel", action="store_true")
    p.add_argument("--detect-prompt", default=os.environ.get("DETECT_PROMPT_FILE", ""),
                   help="text file with your own detection prompt for step 1 ({area} is replaced by the area line)")
    a = p.parse_args()
    from vl_infer import RoomModel

    print("loading the model ...", flush=True)
    model = RoomModel(a.model, adapter=a.adapter)
    ex = sorted(glob.glob(os.path.join(a.examples, "*.jpg")))[:12] if a.examples else []
    prompt = open(a.detect_prompt).read() if a.detect_prompt else None
    app = build(model, ex, prompt)
    if not a.no_tunnel:
        start_tunnel(a.port)
    app.queue().launch(server_name="0.0.0.0", server_port=a.port, allowed_paths=[a.examples or tempfile.gettempdir()])


if __name__ == "__main__":
    main()
