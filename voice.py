from __future__ import annotations

import json
import os
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from queue import Empty, Queue
from typing import Any

import pyautogui
import pygetwindow as gw
import speech_recognition as sr


class VoiceAgent(threading.Thread):
    def __init__(self, shared_state: dict[str, Any]) -> None:
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.commands: Queue[str] = Queue()
        self.running = threading.Event()
        self.running.set()
        self.recognizer = sr.Recognizer()
        try:
            self.microphone = sr.Microphone()
            self.available = True
        except OSError:
            self.microphone = None
            self.available = False
            self.shared_state["voice_status"] = "microphone_occupied"
            self.shared_state["system_status"] = "Microphone occupied"
        except Exception:
            self.microphone = None
            self.available = False
            self.shared_state["voice_status"] = "microphone_unavailable"
            self.shared_state["system_status"] = "Microphone unavailable"
        self.supported_apps = {
            "notepad": ["notepad.exe"],
            "calculator": ["calc.exe"],
            "paint": ["mspaint.exe"],
            "explorer": ["explorer.exe"],
            "browser": ["rundll32.exe", "url.dll,FileProtocolHandler", "https://www.google.com"],
        }

    def enqueue(self, command: str) -> None:
        self.commands.put(command)

    def stop(self) -> None:
        self.running.clear()

    def get_gemini_settings(self) -> tuple[str, str, bool]:
        config = self.shared_state.get("voice_ai", {})
        api_key = str(config.get("api_key") or os.getenv("GEMINI_API_KEY") or "").strip()
        model = str(config.get("model") or "gemini-2.0-flash").strip() or "gemini-2.0-flash"
        enabled = bool(config.get("enabled")) and bool(api_key)
        return api_key, model, enabled

    def execute_open_app(self, app_name: str) -> bool:
        command = self.supported_apps.get(app_name)
        if command is None:
            return False
        subprocess.Popen(command)
        self.shared_state["voice_status"] = f"opened_{app_name}"
        self.shared_state["system_status"] = f"Opened {app_name.title()}"
        return True

    def execute_open_website(self, url: str) -> bool:
        normalized = url.strip()
        if not normalized:
            return False
        if not normalized.startswith(("http://", "https://")):
            normalized = f"https://{normalized}"
        parsed = urllib.parse.urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False
        subprocess.Popen(["rundll32.exe", "url.dll,FileProtocolHandler", normalized])
        self.shared_state["voice_status"] = "opened_website"
        self.shared_state["system_status"] = f"Opened {parsed.netloc}"
        return True

    def execute_key_press(self, key: str) -> bool:
        normalized = key.strip().lower()
        if not normalized:
            return False
        pyautogui.press(normalized)
        self.shared_state["voice_status"] = f"pressed_{normalized}"
        self.shared_state["system_status"] = f"Pressed {normalized}"
        return True

    def execute_hotkey(self, keys: list[str]) -> bool:
        cleaned = [key.strip().lower() for key in keys if key and key.strip()]
        if not cleaned:
            return False
        pyautogui.hotkey(*cleaned)
        self.shared_state["voice_status"] = "hotkey"
        self.shared_state["system_status"] = f"Hotkey: {' + '.join(cleaned)}"
        return True

    def execute_type_text(self, text: str) -> bool:
        if not text:
            return False
        pyautogui.write(text, interval=0.01)
        self.shared_state["voice_status"] = "typed_text"
        self.shared_state["system_status"] = f"Typed: {text[:40]}"
        return True

    def execute_gemini_action(self, action: dict[str, Any]) -> bool:
        action_type = str(action.get("action", "")).strip().lower()
        if action_type == "open_app":
            return self.execute_open_app(str(action.get("app", "")).strip().lower())
        if action_type == "open_website":
            return self.execute_open_website(str(action.get("url", "")))
        if action_type == "press_key":
            return self.execute_key_press(str(action.get("key", "")))
        if action_type == "hotkey":
            keys = action.get("keys", [])
            return self.execute_hotkey(keys if isinstance(keys, list) else [])
        if action_type == "type_text":
            return self.execute_type_text(str(action.get("text", "")))
        if action_type == "take_screenshot":
            screenshot = pyautogui.screenshot()
            screenshot.save("voice_capture.png")
            self.shared_state["voice_status"] = "saved_screenshot"
            self.shared_state["system_status"] = "Saved Screenshot"
            return True
        if action_type == "mute_volume":
            pyautogui.press("volumemute")
            self.shared_state["voice_status"] = "muted_volume"
            self.shared_state["system_status"] = "Muted Volume"
            return True
        if action_type == "maximize_window":
            try:
                active_window = gw.getActiveWindow()
                if active_window is None:
                    return False
                active_window.maximize()
                self.shared_state["voice_status"] = "maximized_window"
                self.shared_state["system_status"] = "Maximized Window"
                return True
            except Exception:
                return False
        if action_type == "clear_canvas":
            self.shared_state["clear_canvas"] = True
            self.shared_state["voice_status"] = "cleared_canvas"
            self.shared_state["system_status"] = "Cleared Canvas"
            return True
        return False

    def plan_with_gemini(self, transcript: str) -> dict[str, Any] | None:
        api_key, model, enabled = self.get_gemini_settings()
        if not enabled:
            return None
        prompt = (
            "You convert a spoken desktop voice command into a safe JSON action.\n"
            "Return JSON only. No markdown.\n"
            "Supported actions are: "
            "open_app, open_website, press_key, hotkey, type_text, take_screenshot, "
            "mute_volume, maximize_window, clear_canvas, no_op.\n"
            "Supported open_app values are: notepad, calculator, paint, explorer, browser.\n"
            "Schema: "
            "{\"action\":\"...\",\"app\":\"...\",\"url\":\"...\",\"key\":\"...\","
            "\"keys\":[\"...\"],\"text\":\"...\",\"reason\":\"...\"}\n"
            "If the request is unsafe, ambiguous, or not supported, return "
            "{\"action\":\"no_op\",\"reason\":\"...\"}.\n"
            f"Voice command: {transcript}"
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        encoded_model = urllib.parse.quote(model, safe="")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{encoded_model}:generateContent?key={urllib.parse.quote(api_key, safe='')}"
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            self.shared_state["voice_status"] = "gemini_error"
            self.shared_state["system_status"] = "Gemini request failed"
            return None
        try:
            text = raw["candidates"][0]["content"]["parts"][0]["text"]
            action = json.loads(text)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            self.shared_state["voice_status"] = "gemini_error"
            self.shared_state["system_status"] = "Gemini response invalid"
            return None
        return action if isinstance(action, dict) else None

    def execute(self, transcript: str) -> None:
        normalized = transcript.lower()
        if "open notepad" in normalized:
            subprocess.Popen(["notepad.exe"])
            self.shared_state["voice_status"] = "opened_notepad"
            self.shared_state["system_status"] = "Opened Notepad"
        elif "open browser" in normalized:
            subprocess.Popen(["rundll32.exe", "url.dll,FileProtocolHandler", "https://www.google.com"])
            self.shared_state["voice_status"] = "opened_browser"
            self.shared_state["system_status"] = "Opened Browser"
        elif "take screenshot" in normalized or "capture screenshot" in normalized:
            screenshot = pyautogui.screenshot()
            screenshot.save("voice_capture.png")
            self.shared_state["voice_status"] = "saved_screenshot"
            self.shared_state["system_status"] = "Saved Screenshot"
        elif "mute volume" in normalized:
            pyautogui.press("volumemute")
            self.shared_state["voice_status"] = "muted_volume"
            self.shared_state["system_status"] = "Muted Volume"
        elif "maximize window" in normalized:
            try:
                active_window = gw.getActiveWindow()
                if active_window is not None:
                    active_window.maximize()
                    self.shared_state["voice_status"] = "maximized_window"
                    self.shared_state["system_status"] = "Maximized Window"
                else:
                    self.shared_state["voice_status"] = "window_not_found"
                    self.shared_state["system_status"] = "No Active Window"
            except Exception:
                self.shared_state["voice_status"] = "window_not_found"
                self.shared_state["system_status"] = "No Active Window"
        elif "clear canvas" in normalized:
            self.shared_state["clear_canvas"] = True
            self.shared_state["voice_status"] = "cleared_canvas"
            self.shared_state["system_status"] = "Cleared Canvas"
        else:
            action = self.plan_with_gemini(transcript)
            if action and str(action.get("action", "")).lower() != "no_op":
                if self.execute_gemini_action(action):
                    return
            reason = ""
            if action and str(action.get("action", "")).lower() == "no_op":
                reason = str(action.get("reason", "")).strip()
            self.shared_state["voice_status"] = f"unmapped:{normalized}"
            self.shared_state["system_status"] = reason or f"Voice: {normalized}"

    def listen_once(self) -> None:
        if not self.available or self.microphone is None:
            return
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio = self.recognizer.listen(
                source,
                timeout=1,
                phrase_time_limit=3,
            )
        transcript = self.recognizer.recognize_google(audio)
        self.execute(transcript)

    def run(self) -> None:
        while self.running.is_set():
            try:
                mode = self.commands.get(timeout=0.2)
            except Empty:
                continue
            if mode == "listen":
                try:
                    self.shared_state["voice_status"] = "listening"
                    self.listen_once()
                except OSError:
                    self.shared_state["voice_status"] = "microphone_occupied"
                    self.shared_state["system_status"] = "Microphone occupied"
                except Exception:
                    self.shared_state["voice_status"] = "voice_error"
                    self.shared_state["system_status"] = "Voice command failed"
