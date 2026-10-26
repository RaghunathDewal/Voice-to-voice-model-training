"""(vision/demo_app.py) Room check demo in the browser: step 1 BEFORE photo -> inventory, step 2 AFTER photo ->
missing / changed / new. v3 counting (single) vs hybrid, plus new items.

    python demo_app.py --adapter /tmp/vl_cmp/v3 [--examples DIR] [--port 7860] [--no-tunnel]
    (on Kaggle: bash run_kaggle.sh demo -> prints "OPEN THIS: https://....trycloudflare.com")

One Qwen3-VL-4B in memory; the v3 LoRA is switched on for counting / boxes and off for "new items" (the base model
lists other objects better: 80% vs 26% in the bench).
  single  the v3 counting answer
  hybrid  the v3 counting answer decides 0 / not 0; if not 0, "locate every X" and count the boxes
Step 1 counts the items on the BEFORE photo and fills the inventory (editable); or type the inventory yourself
("pillow 4, towel 2, lamp 2") and go straight to step 2. New items: objects listed on the AFTER photo and not on the
BEFORE photo (without a BEFORE photo everything is listed).
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


def count_photo(model, img: str, names: list[str], mode: str):
    """single = the v3 counting answer; hybrid = 0 if single says 0, else the number of "locate every X" boxes."""
    single, raw = model.count([img], [names])[0]
    hybrid, boxes = dict(single), {}
    if mode != "single":
        ask = [n for n in names if single.get(n)]                    # only items the count says are there
        for n, (bs, _) in zip(ask, model.locate([(img, n) for n in ask]) if ask else []):
            boxes[n] = bs or []
            if bs is not None:
                hybrid[n] = len(bs)
    return single, hybrid, boxes, raw


def other_objects(model, img: str, names: list[str]) -> dict:
    """Objects NOT on the list (base model, LoRA off), minus room parts and listed items."""
    return clean_new_items(model.new_items([img], [names], base_only=True)[0][0], names)


def count_before(model, before_img, items_text, mode, want_new):
    """Step 1: count the items on the BEFORE photo -> the before inventory (filled into the text box)."""
    t0 = time.time()
    if not before_img:
        return None, [], items_text, {}, "", "Upload a BEFORE photo."
    names, _ = parse_expected(items_text)
    if not names:
        return None, [], items_text, {}, "", "Type the items to count, e.g. pillow, towel, lamp, chair"
    single, hybrid, boxes, raw = count_photo(model, before_img, names, mode)
    use = single if mode == "single" else hybrid                   # what goes into the inventory
    rows = [[n, single.get(n, "?"), hybrid.get(n, "?") if mode != "single" else "-"] for n in names]
    inventory = ", ".join(f"{n} {use[n]}" if use.get(n) is not None else n for n in names)
    others = other_objects(model, before_img, names) if want_new else {}
    others_md = ("**Other objects in the BEFORE photo** (kept to find new items after): "
                 + (", ".join(f"{k} ({v})" for k, v in others.items()) or "none")) if want_new else ""
    info = (f"{time.time() - t0:.1f} s | inventory = the {'single' if mode == 'single' else 'hybrid'} counts "
            f"(edit them if a count is wrong), now upload the AFTER photo\n\nraw count answer: `{raw.strip()[:300]}`")
    return draw_boxes(before_img, boxes) if boxes else None, rows, inventory, others, others_md, info


def check_after(model, after_img, inventory_text, before_others, mode, want_new, tolerance):
    """Step 2: count the AFTER photo against the inventory -> ok / MISSING / fewer / more, and new items."""
    t0 = time.time()
    if not after_img:
        return None, [], "", "Upload an AFTER photo."
    names, before = parse_expected(inventory_text)
    if not names:
        return None, [], "", "No inventory: count a BEFORE photo first (step 1) or type it, e.g. pillow 4, towel 2"
    single, hybrid, boxes, raw = count_photo(model, after_img, names, mode)
    rows = []
    for n in names:
        b = before.get(n)
        rows.append([n, "" if b is None else b, single.get(n, "?"), status(b, single.get(n), tolerance)]
                    + ([hybrid.get(n, "?"), status(b, hybrid.get(n), tolerance)] if mode != "single" else ["-", "-"]))
    new_txt = ""
    if want_new:
        after_new = other_objects(model, after_img, names)
        if before_others:
            fresh = {k: v for k, v in after_new.items() if not any(same_object(k, o) for o in before_others)}
            more = {k: v for k, v in after_new.items() if k in before_others and v > before_others[k]}
            new_txt = ("**New since the BEFORE photo:** " + (", ".join(f"{k} ({v})" for k, v in fresh.items()) or "none")
                       + ("\n\n**More than before:** " + ", ".join(f"{k} {before_others[k]} -> {v}"
                                                                   for k, v in more.items()) if more else ""))
        else:
            new_txt = ("**Other objects in the AFTER photo** (no BEFORE photo counted, so everything is listed): "
                       + (", ".join(f"{k} ({v})" for k, v in after_new.items()) or "none"))
    info = f"{time.time() - t0:.1f} s\n\nraw count answer: `{raw.strip()[:300]}`"
    return draw_boxes(after_img, boxes) if boxes else None, rows, new_txt, info


def build(model, examples: list[str]):
    import gradio as gr

    with gr.Blocks(title="Room check") as app:
        gr.Markdown("## Room check (v3)\n**Step 1** count the BEFORE photo -> inventory. **Step 2** check the AFTER "
                    "photo -> missing / changed / new. You can also skip step 1 and type the inventory.")
        with gr.Row():
            mode = gr.Radio(["single", "hybrid", "both"], value="both", label="Counting",
                            info="single = v3 answer | hybrid = v3 decides 0 / not 0, boxes give how many")
            want_new = gr.Checkbox(value=True, label="New items (base model, LoRA off)")
            tol = gr.Slider(0, 2, value=0, step=1, label="Tolerance (difference still 'ok')")
        items = gr.Textbox(label="Items / inventory", value="pillow, towel, lamp, chair", lines=2,
                           info="step 1 fills the counts in (edit if wrong); or type them: pillow 4, towel 2, lamp 2")
        before_others = gr.State({})
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 1: BEFORE")
                before = gr.Image(type="filepath", label="BEFORE photo")
                go1 = gr.Button("1. Count BEFORE", variant="primary")
                before_boxed = gr.Image(label="Boxes on the BEFORE photo")
                before_table = gr.Dataframe(headers=["item", "single", "hybrid"], label="BEFORE counts")
                before_md = gr.Markdown()
                before_info = gr.Markdown()
            with gr.Column():
                gr.Markdown("### Step 2: AFTER")
                after = gr.Image(type="filepath", label="AFTER photo")
                go2 = gr.Button("2. Check AFTER", variant="primary")
                after_boxed = gr.Image(label="Boxes on the AFTER photo")
                after_table = gr.Dataframe(headers=["item", "before", "single", "single status", "hybrid",
                                                    "hybrid status"], label="AFTER vs inventory", wrap=True)
                after_md = gr.Markdown()
                after_info = gr.Markdown()
        go1.click(lambda *a: count_before(model, *a), [before, items, mode, want_new],
                  [before_boxed, before_table, items, before_others, before_md, before_info])
        go2.click(lambda *a: check_after(model, *a), [after, items, before_others, mode, want_new, tol],
                  [after_boxed, after_table, after_md, after_info])
        before.clear(lambda: {}, None, before_others)                  # a new room: forget the old "other objects"
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
