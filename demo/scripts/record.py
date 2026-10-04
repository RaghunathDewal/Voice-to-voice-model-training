import json, subprocess, time, os, shutil
from playwright.sync_api import sync_playwright
OUT = "/tmp/rec"; shutil.rmtree(f"{OUT}/video", ignore_errors=True); os.makedirs(f"{OUT}/shots", exist_ok=True)
srv = subprocess.Popen(["python", "-m", "http.server", "8765", "-d", "/tmp/site"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
steps, clicks = [], []
with sync_playwright() as pw:
    b = pw.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
                           args=["--autoplay-policy=no-user-gesture-required"])
    ctx = b.new_context(viewport={"width": 1920, "height": 1080}, record_video_dir=f"{OUT}/video",
                        record_video_size={"width": 1920, "height": 1080})
    t_ref = time.time()
    page = ctx.new_page()
    errors = []; page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("http://127.0.0.1:8765/?demo=demo"); page.wait_for_load_state("networkidle"); page.wait_for_timeout(1500)
    now = lambda: round(time.time() - t_ref, 3)
    def shot(name, caption):
        page.screenshot(path=f"{OUT}/shots/{name}.png"); steps.append({"name": name, "t": now(), "caption": caption})
    def click(sel, label):
        box = page.locator(sel).bounding_box(); clicks.append({"selector": sel, "label": label, "t": now(), "box": box})
        page.click(sel)
    shot("01_landing", "The voice concierge, ready")
    click("#tab-model", "Model tab"); page.wait_for_timeout(1200); shot("02_model", "1.3B speech-to-speech LLM")
    click("#tab-booking", "Booking tab"); page.wait_for_timeout(900)
    click("#start", "Start talking"); page.wait_for_timeout(700); shot("03_listening", "Guest starts talking")
    page.wait_for_function("window.__events.some(e => e.type==='bot_start' && e.turn===0)", timeout=60000); page.wait_for_timeout(2500)
    shot("04_greeting", "Concierge introduces itself")
    page.wait_for_selector("#tool-0", timeout=60000); page.wait_for_timeout(600); shot("05_tool_order", "Tool call: order 2 towels")
    page.wait_for_function("window.__events.some(e => e.type==='bot_start' && e.turn===1)", timeout=60000); page.wait_for_timeout(1500)
    click("#tool-0", "Order details"); page.wait_for_timeout(1200); shot("06_tool_details", "Tool call details")
    page.wait_for_selector("#tool-1", timeout=60000); page.wait_for_timeout(600); shot("07_issue", "Maintenance ticket created")
    page.wait_for_function("window.__events.some(e => e.type==='bot_start' && e.turn===3)", timeout=60000); page.wait_for_timeout(2500)
    shot("08_breakfast", "Hotel info: breakfast hours")
    page.wait_for_function("window.__events.some(e => e.type==='done')", timeout=90000); page.wait_for_timeout(600)
    shot("09_room", "Answers from the booking")
    click("#end", "End call"); page.wait_for_timeout(1500); shot("10_end", "Call ended")
    origin = page.evaluate("performance.timeOrigin")
    events = page.evaluate("window.__events")
    for e in events: e["t"] = round((origin + e["t"]) / 1000 - t_ref, 3)
    video_path = page.video.path(); ctx.close(); b.close()
srv.terminate()
json.dump({"video": video_path, "steps": steps, "clicks": clicks, "events": events, "errors": errors}, open(f"{OUT}/session.json", "w"), indent=1)
print("video", video_path, "errors", errors)
print([(e["type"], e.get("turn"), e["t"]) for e in events if e["type"] in ("guest_start", "bot_start", "bot_end", "done", "start")])
