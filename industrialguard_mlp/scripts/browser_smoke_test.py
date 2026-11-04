"""Check the live Streamlit UI in an isolated headless Chrome session."""
from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from websockets.sync.client import connect


ROOT = Path(__file__).resolve().parents[1]
URL = os.environ.get("INDUSTRIALGUARD_URL", "http://127.0.0.1:8765/")
RESULT_PATH = ROOT / "reports" / "browser_smoke_test.json"
SCREENSHOT_PATH = ROOT / "reports" / "browser_demo.png"
PIC2_SCREENSHOT_PATH = ROOT / "reports" / "pic2_browser_result.png"
CDP = "http://127.0.0.1:9222"
PROFILE = None
CASES = [
    ("normal_cable", ROOT / "datasets/mvtec_ad/cable/test/good/000.png", "Identified as Cable (industrial mlp)"),
    ("defective_cable", ROOT / "datasets/mvtec_ad/cable/test/bent_wire/000.png", "Identified as Cable (industrial mlp)"),
    ("capsule", ROOT / "datasets/mvtec_ad/capsule/test/good/000.png", "Identified as Capsule (industrial mlp)"),
    ("person_photo", Path(r"C:\Users\princ\Downloads\me.jpg"), "Unsupported or uncertain product image."),
    ("consumer_charging_cable", Path(r"D:\pic2.jpg"), "Identified as Cable"),
]


def start_chrome() -> subprocess.Popen | None:
    global PROFILE
    try:
        urllib.request.urlopen(f"{CDP}/json/version", timeout=1).close()
        raise RuntimeError("Chrome DevTools port 9222 is already in use; refusing to control that browser.")
    except urllib.error.URLError:
        pass
    chrome = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    if not chrome.exists():
        raise FileNotFoundError(f"Chrome executable was not found at {chrome}")
    PROFILE = tempfile.TemporaryDirectory(prefix="industrialguard_browser_")
    profile = Path(PROFILE.name)
    flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    process = subprocess.Popen([
        str(chrome), "--headless=new", "--no-first-run", "--disable-gpu",
        "--no-default-browser-check", "--remote-debugging-port=9222",
        f"--user-data-dir={profile}", URL,
    ], creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{CDP}/json/version", timeout=1).close()
            return process
        except urllib.error.URLError:
            time.sleep(.25)
    process.terminate()
    raise RuntimeError("Headless Chrome did not start its DevTools endpoint.")


class Browser:
    def __init__(self):
        pages = json.load(urllib.request.urlopen(f"{CDP}/json", timeout=5))
        page = next(p for p in pages if p.get("type") == "page" and URL.rstrip("/") in p.get("url", ""))
        self.ws = connect(page["webSocketDebuggerUrl"], open_timeout=10, legacy=True)
        self.next_id = 0
        self.call("Page.enable")
        self.reset_page()

    def reset_page(self):
        self.call("Page.navigate", {"url": URL})
        deadline = time.time() + 45
        while time.time() < deadline:
            loaded = self.evaluate("document.querySelector('input[type=file]') && "
                                   "[...document.querySelectorAll('button')].some(x=>x.innerText.trim()==='Analyze images')")
            if loaded:
                return
            time.sleep(.5)
        raise RuntimeError("IndustrialGuard AI upload controls did not load in Chrome.")

    def call(self, method, params=None):
        self.next_id += 1
        call_id = self.next_id
        self.ws.send(json.dumps({"id": call_id, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.ws.recv(timeout=30))
            if message.get("id") == call_id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})

    def evaluate(self, expression):
        answer = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True,
                                                 "awaitPromise": True, "userGesture": True})
        return answer.get("result", {}).get("value")

    def text(self):
        return self.evaluate("document.body.innerText") or ""

    def click_at(self, x: float, y: float):
        self.call("Input.dispatchMouseEvent", {"type": "mousePressed", "x": x, "y": y,
                                                "button": "left", "clickCount": 1})
        self.call("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": x, "y": y,
                                                "button": "left", "clickCount": 1})

    def click_button(self, label: str) -> bool:
        serialized = json.dumps(label)
        point = self.evaluate(
            "(() => { const e=[...document.querySelectorAll('button')].find(x=>x.innerText.trim()==="
            f"{serialized} && !x.disabled); if(!e) return null; e.scrollIntoView({{block:'center'}}); "
            "const r=e.getBoundingClientRect(); "
            "return JSON.stringify({x:r.x+r.width/2,y:r.y+r.height/2}); })()")
        if not point:
            return False
        self.click_at(**json.loads(point))
        return True

    def upload(self, paths: list[Path]):
        doc = self.call("DOM.getDocument", {"depth": -1, "pierce": True})
        found = self.call("DOM.querySelector", {"nodeId": doc["root"]["nodeId"],
                                                "selector": "input[type='file']"})
        if not found.get("nodeId"):
            raise RuntimeError("Image upload control was not rendered.")
        self.call("DOM.setFileInputFiles", {"files": [str(path) for path in paths], "nodeId": found["nodeId"]})
        deadline = time.time() + 30
        while time.time() < deadline:
            ready = self.evaluate("(() => { const b=[...document.querySelectorAll('button')].find(x=>x.innerText.trim()==='Analyze images'); return !!b && !b.disabled; })()")
            if ready:
                return
            time.sleep(.25)
        raise RuntimeError("The uploaded image(s) did not reach the Analyze images button.")

    def analyze(self):
        if not self.click_button("Analyze images"):
            raise RuntimeError("Could not click Analyze images.")
        time.sleep(1)

    def wait_for(self, expected: list[str], timeout: int = 25) -> str:
        deadline = time.time() + timeout
        latest = ""
        while time.time() < deadline:
            latest = self.text()
            if all(item in latest for item in expected):
                return latest
            time.sleep(.5)
        return latest

    def select_category(self, wanted: str) -> bool:
        found_at = None
        point = self.evaluate("(() => { const e=document.querySelectorAll('[role=combobox]')[1]; if(!e) return null; "
                              "e.scrollIntoView({block:'center'}); const r=e.getBoundingClientRect(); "
                              "return JSON.stringify({x:r.x+r.width/2,y:r.y+r.height/2}); })()")
        if not point:
            return False
        self.click_at(**json.loads(point))
        time.sleep(.25)
        display_name = wanted.replace("_", " ").lower()
        needle = json.dumps(display_name)
        option_point = self.evaluate("(() => { const e=[...document.querySelectorAll('[role=option]')].find(x=>x.innerText.trim().toLowerCase()==="
                                      f"{needle}); if(!e) return null; e.scrollIntoView({{block:'center'}}); "
                                      "const r=e.getBoundingClientRect(); "
                                      "return JSON.stringify({x:r.x+r.width/2,y:r.y+r.height/2}); })()")
        for fraction in (.25, .5, .75, 1.0):
            if option_point:
                break
            self.evaluate("(() => { const root=document.querySelector('[role=listbox]'); if(root) "
                          f"root.scrollTop=(root.scrollHeight-root.clientHeight)*{fraction}; }})()")
            time.sleep(.18)
            option_point = self.evaluate("(() => { const e=[...document.querySelectorAll('[role=option]')].find(x=>x.innerText.trim().toLowerCase()==="
                                         f"{needle}); if(!e) return null; e.scrollIntoView({{block:'center'}}); "
                                         "const r=e.getBoundingClientRect(); "
                                         "return JSON.stringify({x:r.x+r.width/2,y:r.y+r.height/2}); })()")
            if option_point:
                found_at = fraction
        click_point = json.loads(option_point) if option_point else None
        click_target = (self.evaluate(
            f"(() => {{const e=document.elementFromPoint({click_point['x']},{click_point['y']}); "
            "return e&&{text:e.innerText,role:e.getAttribute('role'),tag:e.tagName};})()")
                        if click_point else None)
        if option_point:
            self.click_at(**json.loads(option_point))
            time.sleep(.2)
        marker = ("You chose Other product manually" if display_name.startswith("other product")
                  else f"Identified as {display_name.title()} (manual selection)")
        deadline = time.time() + 10
        while time.time() < deadline:
            page = self.text()
            if marker in page or (not display_name.startswith("other product")
                                  and "You selected this category manually" in page):
                self.last_selection_debug = {"wanted": wanted, "option_found_at": found_at,
                                             "click_point": click_point,
                                             "click_target": click_target,
                                             "marker": marker, "combobox": self.evaluate("document.querySelectorAll('[role=combobox]')[1]?.innerText"),
                                             "matched": True}
                return True
            time.sleep(.25)
        if display_name.startswith("other product"):
            # The final option can sit below the menu's visible scroll area.
            # Select it with the menu keyboard controls if the click did not stick.
            self.click_at(**json.loads(point))
            for event_type in ("keyDown", "keyUp"):
                self.call("Input.dispatchKeyEvent", {"type": event_type, "key": "End",
                                                      "code": "End", "windowsVirtualKeyCode": 35})
            for event_type in ("keyDown", "keyUp"):
                self.call("Input.dispatchKeyEvent", {"type": event_type, "key": "Enter",
                                                      "code": "Enter", "windowsVirtualKeyCode": 13})
            deadline = time.time() + 10
            while time.time() < deadline:
                if marker in self.text():
                    return True
                time.sleep(.25)
        else:
            # Long Streamlit select menus virtualize off-screen rows. Use the
            # category index as a keyboard fallback if the visible click did
            # not change the selected value.
            try:
                categories = json.loads((ROOT / "models" / "category_index.json").read_text(
                    encoding="utf-8"))["categories"]
                wanted_key = display_name.replace(" ", "_")
                option_index = categories.index(wanted_key) + 1  # automatic mode is option zero
            except (OSError, ValueError, KeyError):
                option_index = None
            if option_index is not None:
                if not self.evaluate("!!document.querySelector('[role=listbox]')"):
                    self.click_at(**json.loads(point))
                    time.sleep(.2)
                for key, code, virtual_key in [("Home", "Home", 36)] + [
                        ("ArrowDown", "ArrowDown", 40)] * option_index + [("Enter", "Enter", 13)]:
                    for event_type in ("keyDown", "keyUp"):
                        self.call("Input.dispatchKeyEvent", {"type": event_type, "key": key,
                                                              "code": code, "windowsVirtualKeyCode": virtual_key})
                deadline = time.time() + 10
                while time.time() < deadline:
                    page = self.text()
                    if marker in page or "You selected this category manually" in page:
                        self.last_selection_debug = {"wanted": wanted,
                                                     "option_found_at": found_at,
                                                     "click_point": click_point,
                                                     "keyboard_fallback": True,
                                                     "matched": True}
                        return True
                    time.sleep(.25)
        page = self.text()
        self.last_selection_debug = {"wanted": wanted, "option_found_at": found_at,
                                     "click_point": click_point,
                                     "click_target": click_target,
                                     "marker": marker, "combobox": self.evaluate("document.querySelectorAll('[role=combobox]')[1]?.innerText"),
                                     "matched": False,
                                     "page_lines": [line for line in page.splitlines()
                                                    if "Identified as" in line or "manually" in line.lower()]}
        return False

    def screenshot(self, path: Path = SCREENSHOT_PATH):
        self.call("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1200,
                                                         "deviceScaleFactor": 1, "mobile": False})
        self.evaluate("document.activeElement && document.activeElement.blur(); window.scrollTo(0,0); "
                       "document.documentElement.scrollTop=0; document.body.scrollTop=0; "
                       "[...document.querySelectorAll('*')].forEach(e=>{if(e.scrollTop)e.scrollTop=0})")
        time.sleep(1)
        capture = self.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
        path.write_bytes(base64.b64decode(capture["data"]))


def main():
    process = start_chrome()
    browser = None
    results = []
    try:
        browser = Browser()
        for name, path, expected in CASES:
            if not path.exists():
                results.append({"case": name, "status": "SKIP", "reason": f"missing image: {path}"})
                continue
            print(f"Browser test: {name}", flush=True)
            if results:
                browser.reset_page()
            browser.upload([path])
            if "Identified as" in browser.text() or "Unsupported or uncertain" in browser.text():
                raise RuntimeError("Results appeared before the Analyze button was pressed.")
            browser.analyze()
            page = browser.wait_for([expected])
            results.append({"case": name, "status": "PASS" if expected in page else "FAIL",
                            "expected_text": expected,
                            "observed_lines": [line for line in page.splitlines()
                                               if "Identified as" in line or "Unsupported or uncertain" in line
                                               or "Damage screen score" in line
                                               or "Reference difference" in line
                                               or "Anomaly score" in line]})
            if name == "normal_cable":
                browser.screenshot()
            elif name == "consumer_charging_cable":
                browser.screenshot(PIC2_SCREENSHOT_PATH)

        normal_cable = CASES[0][1]
        capsule = CASES[2][1]
        print("Browser test: multi_image_batch", flush=True)
        browser.reset_page()
        browser.upload([normal_cable, capsule])
        browser.analyze()
        page = browser.wait_for(["Identified as Cable (industrial mlp)",
                                 "Identified as Capsule (industrial mlp)"])
        batch_pass = ("Identified as Cable (industrial mlp)" in page
                      and "Identified as Capsule (industrial mlp)" in page)
        results.append({"case": "multi_image_batch", "status": "PASS" if batch_pass else "FAIL",
                        "expected_count": 2, "observed_lines": [line for line in page.splitlines()
                                                                  if "Identified as" in line]})

        consumer = Path(r"D:\pic2.jpg")
        if consumer.exists():
            print("Browser test: manual_category_override", flush=True)
            browser.reset_page()
            browser.upload([consumer])
            browser.analyze()
            page = browser.wait_for(["Identified as Cable"])
            selected = browser.select_category("cable")
            manual_page = browser.wait_for(["You selected this category manually"])
            passed = selected and "You selected this category manually" in manual_page
            results.append({"case": "manual_category_override", "status": "PASS" if passed else "FAIL",
                            "observed_text": [line for line in manual_page.splitlines()
                                              if "Identified as" in line or "manually" in line.lower()]})
            # Exercise the custom-product fallback in a fresh analyzed item so
            # the previous per-image manual category cannot affect this case.
            browser.reset_page()
            browser.upload([consumer])
            browser.analyze()
            other_selected = browser.select_category("Other product (manual)")
            other_page = browser.wait_for(["You chose Other product manually"])
            other_pass = other_selected and "You chose Other product manually" in other_page
            results.append({"case": "other_product_manual_damage_screen",
                            "status": "PASS" if other_pass else "FAIL",
                            "observed_lines": [line for line in other_page.splitlines()
                                               if "Other product" in line or "exploratory" in line.lower()
                                               or "candidate visual region" in line.lower()]})

        bottle_samples = sorted((ROOT / "datasets/additional/Plastic Shampoo Bottle Dataset/test/images").glob("*.jpg"))
        if bottle_samples:
            print("Browser test: plastic_shampoo_bottle_automatic_category", flush=True)
            browser.reset_page()
            browser.upload([bottle_samples[0]])
            browser.analyze()
            bottle_page = browser.wait_for(["Identified as"])
            automatic_ok = "Identified as Plastic Shampoo Bottle (industrial mlp)" in bottle_page
            results.append({"case": "plastic_shampoo_bottle_automatic_category",
                            "status": "PASS" if automatic_ok else "FAIL",
                            "expected_text": "Identified as Plastic Shampoo Bottle (industrial mlp)",
                            "observed_lines": [line for line in bottle_page.splitlines()
                                               if "Identified as" in line or "Unsupported" in line]})

            print("Browser test: bottle_specific_yolo_manual_override", flush=True)
            selected = browser.select_category("plastic_shampoo_bottle")
            bottle_page = browser.wait_for(["Bottle YOLO candidate detections"])
            yolo_ok = selected and "Bottle YOLO candidate detections" in bottle_page
            results.append({"case": "bottle_specific_yolo_manual_override",
                            "status": "PASS" if yolo_ok else "FAIL",
                            "yolo_report_rendered": yolo_ok,
                            "observed_lines": [line for line in bottle_page.splitlines()
                                               if "Identified as" in line or "YOLO" in line
                                               or "candidate" in line.lower()]})

        report = {"browser": "isolated headless Chrome via CDP", "url": URL,
                  "page_rendered": browser is not None, "cases": results,
                  "screenshot": str(SCREENSHOT_PATH),
                  "consumer_charging_cable_screenshot": str(PIC2_SCREENSHOT_PATH)}
        RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        if any(item["status"] == "FAIL" for item in results):
            raise SystemExit(1)
    finally:
        if browser is not None:
            try:
                browser.call("Browser.close")
            except Exception:
                pass
            browser.ws.close()
        if process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
        if PROFILE is not None:
            PROFILE.cleanup()


if __name__ == "__main__":
    main()
