from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any, Callable


class UserPanel:
    PRESETS: dict[str, dict[str, float | int]] = {
        "Precision": {
            "click_pinch_threshold": 0.048,
            "draw_pinch_threshold": 0.044,
            "dead_zone_radius": 0.045,
            "confirmation_window": 6,
            "drag_activation_frames": 8,
            "drag_release_confirmation_frames": 4,
            "canvas_alpha": 0.24,
            "scroll_sensitivity": 520.0,
        },
        "Balanced": {
            "click_pinch_threshold": 0.055,
            "draw_pinch_threshold": 0.05,
            "dead_zone_radius": 0.035,
            "confirmation_window": 5,
            "drag_activation_frames": 7,
            "drag_release_confirmation_frames": 3,
            "canvas_alpha": 0.32,
            "scroll_sensitivity": 700.0,
        },
        "Fast": {
            "click_pinch_threshold": 0.068,
            "draw_pinch_threshold": 0.06,
            "dead_zone_radius": 0.02,
            "confirmation_window": 3,
            "drag_activation_frames": 5,
            "drag_release_confirmation_frames": 2,
            "canvas_alpha": 0.48,
            "scroll_sensitivity": 980.0,
        },
    }

    def __init__(
        self,
        shared_state: dict[str, Any],
        voice_trigger: Callable[[], None],
        config_path: str = "config.json",
    ) -> None:
        self.shared_state = shared_state
        self.voice_trigger = voice_trigger
        self.config_path = Path(config_path)
        self.root = tk.Tk()
        self.root.title("Virtual Mouse Control Panel")
        self.root.geometry("500x760")
        self.root.minsize(500, 760)
        self.scroll_var = tk.DoubleVar(value=700.0)
        self.click_threshold_var = tk.DoubleVar(value=0.055)
        self.draw_threshold_var = tk.DoubleVar(value=0.05)
        self.dead_zone_var = tk.DoubleVar(value=0.035)
        self.confirmation_window_var = tk.IntVar(value=5)
        self.drag_activation_var = tk.IntVar(value=7)
        self.drag_release_var = tk.IntVar(value=3)
        self.canvas_alpha_var = tk.DoubleVar(value=0.32)
        self.preset_var = tk.StringVar(value="Balanced")
        self.gesture_vars = {
            "left_click": tk.StringVar(value="thumb_index_pinch"),
            "right_click": tk.StringVar(value="thumb_middle_pinch"),
            "draw_toggle": tk.StringVar(value="thumb_ring_pinch"),
        }
        self.status_var = tk.StringVar(value="idle")
        self.profile_var = tk.StringVar(value="default")
        self.voice_var = tk.StringVar(value="standby")
        self.system_var = tk.StringVar(value="ready")
        self.gemini_enabled_var = tk.BooleanVar(value=False)
        self.gemini_api_key_var = tk.StringVar(value="")
        self.gemini_model_var = tk.StringVar(value="gemini-2.0-flash")
        self.build()
        self.load_config()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.schedule_refresh()

    def build(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)
        action_frame = ttk.LabelFrame(container, text="Quick Actions", padding=12)
        action_frame.pack(fill="x", pady=(0, 14))
        ttk.Button(
            action_frame,
            text="Start Voice Command",
            command=self.trigger_voice,
        ).pack(fill="x", pady=(0, 8))
        ttk.Label(
            action_frame,
            text="Hand gesture: show index and middle fingers only for a moment.",
            wraplength=440,
            justify="left",
        ).pack(anchor="w")
        ttk.Label(container, text="Available Gestures").pack(anchor="w")
        gesture_list = tk.Listbox(container, height=6)
        for label in [
            "move",
            "left_click  - thumb + index pinch",
            "right_click - thumb + middle pinch",
            "scroll      - four fingers extended",
            "draw_toggle - thumb + ring pinch",
            "voice       - index + middle only",
        ]:
            gesture_list.insert("end", label)
        gesture_list.configure(state="disabled")
        gesture_list.pack(fill="x", pady=(6, 14))
        ttk.Label(container, text="Scroll Sensitivity").pack(anchor="w")
        ttk.Scale(
            container,
            from_=200.0,
            to=1500.0,
            variable=self.scroll_var,
            command=self.on_scroll_change,
        ).pack(fill="x", pady=(6, 14))
        preset_frame = ttk.LabelFrame(container, text="Presets", padding=12)
        preset_frame.pack(fill="x", pady=(0, 14))
        ttk.Combobox(
            preset_frame,
            textvariable=self.preset_var,
            values=tuple(self.PRESETS.keys()),
            state="readonly",
        ).pack(fill="x", pady=(0, 10))
        ttk.Button(
            preset_frame,
            text="Apply Preset",
            command=self.apply_preset,
        ).pack(fill="x")
        calibration_frame = ttk.LabelFrame(container, text="Calibration", padding=12)
        calibration_frame.pack(fill="x", pady=(0, 14))
        self.build_scale(calibration_frame, "Click Pinch", self.click_threshold_var, 0.02, 0.12)
        self.build_scale(calibration_frame, "Draw Pinch", self.draw_threshold_var, 0.02, 0.12)
        self.build_scale(calibration_frame, "Dead Zone", self.dead_zone_var, 0.01, 0.08)
        self.build_scale(calibration_frame, "Canvas Smooth", self.canvas_alpha_var, 0.1, 0.7)
        self.build_scale(calibration_frame, "Confirm Frames", self.confirmation_window_var, 3, 8)
        self.build_scale(calibration_frame, "Drag Hold", self.drag_activation_var, 5, 12)
        self.build_scale(calibration_frame, "Drag Release", self.drag_release_var, 2, 6)
        mapping_frame = ttk.LabelFrame(container, text="Gesture Remap", padding=12)
        mapping_frame.pack(fill="x", pady=(0, 14))
        choices = (
            "thumb_index_pinch",
            "thumb_middle_pinch",
            "thumb_ring_pinch",
        )
        for action_name, variable in self.gesture_vars.items():
            row = ttk.Frame(mapping_frame)
            row.pack(fill="x", pady=4)
            ttk.Label(row, text=action_name, width=16).pack(side="left")
            ttk.Combobox(
                row,
                textvariable=variable,
                values=choices,
                state="readonly",
            ).pack(side="left", fill="x", expand=True)
        ttk.Button(
            container,
            text="Apply Calibration",
            command=self.apply_calibration,
        ).pack(fill="x", pady=(0, 10))
        ttk.Button(
            container,
            text="Apply Gesture Map",
            command=self.apply_gesture_map,
        ).pack(fill="x", pady=(0, 10))
        voice_ai_frame = ttk.LabelFrame(container, text="Voice AI", padding=12)
        voice_ai_frame.pack(fill="x", pady=(0, 14))
        ttk.Checkbutton(
            voice_ai_frame,
            text="Enable Gemini planner",
            variable=self.gemini_enabled_var,
            command=self.apply_voice_ai_settings,
        ).pack(anchor="w", pady=(0, 8))
        ttk.Label(
            voice_ai_frame,
            text="API key is optional in this file if GEMINI_API_KEY is already set in your environment.",
            wraplength=440,
            justify="left",
        ).pack(anchor="w", pady=(0, 8))
        ttk.Entry(
            voice_ai_frame,
            textvariable=self.gemini_api_key_var,
            show="*",
        ).pack(fill="x", pady=(0, 8))
        ttk.Entry(
            voice_ai_frame,
            textvariable=self.gemini_model_var,
        ).pack(fill="x", pady=(0, 8))
        ttk.Button(
            voice_ai_frame,
            text="Apply Voice AI Settings",
            command=self.apply_voice_ai_settings,
        ).pack(fill="x")
        status_frame = ttk.LabelFrame(container, text="Runtime Status", padding=12)
        status_frame.pack(fill="x")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.profile_var).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.voice_var).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.system_var).pack(anchor="w")

    def load_config(self) -> None:
        if not self.config_path.exists():
            self.save_config()
            return
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.save_config()
            return
        self.scroll_var.set(float(data.get("scroll_sensitivity", 700.0)))
        self.preset_var.set(str(data.get("preset", "Balanced")))
        self.click_threshold_var.set(float(data.get("click_pinch_threshold", 0.055)))
        self.draw_threshold_var.set(float(data.get("draw_pinch_threshold", 0.05)))
        self.dead_zone_var.set(float(data.get("dead_zone_radius", 0.035)))
        self.confirmation_window_var.set(int(data.get("confirmation_window", 5)))
        self.drag_activation_var.set(int(data.get("drag_activation_frames", 7)))
        self.drag_release_var.set(int(data.get("drag_release_confirmation_frames", 3)))
        self.canvas_alpha_var.set(float(data.get("canvas_alpha", 0.32)))
        gesture_map = data.get("gesture_map", {})
        for key, variable in self.gesture_vars.items():
            variable.set(gesture_map.get(key, variable.get()))
        voice_ai = data.get("voice_ai", {})
        self.gemini_enabled_var.set(bool(voice_ai.get("enabled", False)))
        self.gemini_api_key_var.set(str(voice_ai.get("api_key", "")))
        self.gemini_model_var.set(str(voice_ai.get("model", "gemini-2.0-flash")))
        self.shared_state["scroll_sensitivity"] = self.scroll_var.get()
        self.shared_state["gesture_map"] = {
            key: variable.get() for key, variable in self.gesture_vars.items()
        }
        self.shared_state["calibration"] = self.collect_calibration()
        self.shared_state["voice_ai"] = self.collect_voice_ai_settings()

    def save_config(self) -> None:
        payload = {
            "preset": self.preset_var.get(),
            "scroll_sensitivity": self.scroll_var.get(),
            "click_pinch_threshold": self.click_threshold_var.get(),
            "draw_pinch_threshold": self.draw_threshold_var.get(),
            "dead_zone_radius": self.dead_zone_var.get(),
            "confirmation_window": self.confirmation_window_var.get(),
            "drag_activation_frames": self.drag_activation_var.get(),
            "drag_release_confirmation_frames": self.drag_release_var.get(),
            "canvas_alpha": self.canvas_alpha_var.get(),
            "gesture_map": {
                key: variable.get() for key, variable in self.gesture_vars.items()
            },
            "voice_ai": self.collect_voice_ai_settings(),
        }
        self.config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def on_scroll_change(self, _: str) -> None:
        self.shared_state["scroll_sensitivity"] = self.scroll_var.get()
        self.save_config()

    def collect_calibration(self) -> dict[str, float | int]:
        return {
            "click_pinch_threshold": float(self.click_threshold_var.get()),
            "draw_pinch_threshold": float(self.draw_threshold_var.get()),
            "dead_zone_radius": float(self.dead_zone_var.get()),
            "confirmation_window": int(self.confirmation_window_var.get()),
            "drag_activation_frames": int(self.drag_activation_var.get()),
            "drag_release_confirmation_frames": int(self.drag_release_var.get()),
            "canvas_alpha": float(self.canvas_alpha_var.get()),
        }

    def apply_calibration(self) -> None:
        self.shared_state["calibration"] = self.collect_calibration()
        self.shared_state["system_status"] = "Calibration updated"
        self.save_config()

    def apply_preset(self) -> None:
        preset = self.PRESETS.get(self.preset_var.get(), self.PRESETS["Balanced"])
        self.scroll_var.set(float(preset["scroll_sensitivity"]))
        self.click_threshold_var.set(float(preset["click_pinch_threshold"]))
        self.draw_threshold_var.set(float(preset["draw_pinch_threshold"]))
        self.dead_zone_var.set(float(preset["dead_zone_radius"]))
        self.confirmation_window_var.set(int(preset["confirmation_window"]))
        self.drag_activation_var.set(int(preset["drag_activation_frames"]))
        self.drag_release_var.set(int(preset["drag_release_confirmation_frames"]))
        self.canvas_alpha_var.set(float(preset["canvas_alpha"]))
        self.shared_state["scroll_sensitivity"] = self.scroll_var.get()
        self.shared_state["calibration"] = self.collect_calibration()
        self.shared_state["system_status"] = f"Preset: {self.preset_var.get()}"
        self.save_config()

    def apply_gesture_map(self) -> None:
        self.shared_state["gesture_map"] = {
            key: variable.get() for key, variable in self.gesture_vars.items()
        }
        self.shared_state["system_status"] = "Gesture map updated"
        self.save_config()

    def collect_voice_ai_settings(self) -> dict[str, str | bool]:
        return {
            "enabled": bool(self.gemini_enabled_var.get()),
            "api_key": self.gemini_api_key_var.get().strip(),
            "model": self.gemini_model_var.get().strip() or "gemini-2.0-flash",
        }

    def apply_voice_ai_settings(self) -> None:
        self.shared_state["voice_ai"] = self.collect_voice_ai_settings()
        if self.shared_state["voice_ai"]["enabled"]:
            self.shared_state["system_status"] = "Gemini voice planner enabled"
        else:
            self.shared_state["system_status"] = "Gemini voice planner disabled"
        self.save_config()

    def trigger_voice(self) -> None:
        self.voice_trigger()

    def build_scale(
        self,
        parent: ttk.LabelFrame,
        label: str,
        variable: tk.DoubleVar | tk.IntVar,
        minimum: float,
        maximum: float,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, width=14).pack(side="left")
        ttk.Scale(
            row,
            from_=minimum,
            to=maximum,
            variable=variable,
            command=self.on_calibration_change,
        ).pack(side="left", fill="x", expand=True)

    def schedule_refresh(self) -> None:
        gesture = self.shared_state.get("gesture", "idle")
        profile = self.shared_state.get("active_profile", "default")
        voice_status = self.shared_state.get("voice_status", "standby")
        system_status = self.shared_state.get("system_status", "ready")
        self.status_var.set(f"Gesture: {gesture}")
        self.profile_var.set(f"Profile: {profile}")
        self.voice_var.set(f"Voice: {voice_status}")
        self.system_var.set(f"Status: {system_status}")
        self.root.after(100, self.schedule_refresh)

    def on_calibration_change(self, _: str) -> None:
        self.shared_state["calibration"] = self.collect_calibration()
        self.save_config()

    def close(self) -> None:
        self.shared_state["running"] = False
        self.save_config()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
