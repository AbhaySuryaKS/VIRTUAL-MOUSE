from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np

from gestures import GestureController, HandTracker, ProjectionEngine3D, detect_active_profile
from ui import UserPanel
from voice import VoiceAgent


class VirtualMouseApplication:
    def __init__(self) -> None:
        self.shared_state: dict[str, Any] = {
            "running": True,
            "gesture": "idle",
            "active_profile": "default",
            "voice_status": "standby",
            "system_status": "ready",
            "scroll_sensitivity": 700.0,
            "calibration": {
                "click_pinch_threshold": 0.055,
                "draw_pinch_threshold": 0.05,
                "dead_zone_radius": 0.035,
                "confirmation_window": 5,
                "drag_activation_frames": 7,
                "drag_release_confirmation_frames": 3,
                "canvas_alpha": 0.32,
            },
            "gesture_map": {
                "left_click": "thumb_index_pinch",
                "right_click": "thumb_middle_pinch",
                "draw_toggle": "thumb_ring_pinch",
            },
            "aircanvas_points": [],
            "aircanvas_enabled": False,
            "clear_canvas": False,
            "request_voice_command": False,
            "voice_ai": {
                "enabled": False,
                "api_key": "",
                "model": "gemini-2.0-flash",
            },
        }
        self.tracker = HandTracker()
        self.controller = GestureController()
        self.projection = ProjectionEngine3D()
        self.voice_agent = VoiceAgent(self.shared_state)
        self.panel = UserPanel(self.shared_state, self.trigger_voice_command)
        self.frame_counter = 0

    def trigger_voice_command(self) -> None:
        self.voice_agent.enqueue("listen")

    def overlay_aircanvas(self, frame: np.ndarray) -> None:
        points = self.shared_state.get("aircanvas_points", [])
        if not points:
            return
        frame_height, frame_width = frame.shape[:2]
        previous = None
        for point in points:
            if point is None:
                previous = None
                continue
            point_x = int((point[0] / max(self.controller.screen_width, 1)) * frame_width)
            point_y = int((point[1] / max(self.controller.screen_height, 1)) * frame_height)
            current = (point_x, point_y)
            if previous is not None:
                cv2.line(frame, previous, current, (0, 120, 255), 2)
            previous = current

    def vision_loop(self) -> None:
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            self.shared_state["system_status"] = "Camera not found"
            capture.release()
            return
        start_time = time.perf_counter()
        while self.shared_state.get("running", False):
            success, frame = capture.read()
            if not success:
                self.shared_state["system_status"] = "Camera frame unavailable"
                time.sleep(0.01)
                continue
            self.frame_counter += 1
            if self.frame_counter % 30 == 0:
                active_profile = detect_active_profile()
                self.shared_state["active_profile"] = active_profile
                self.controller.set_cursor_precision(
                    active_profile in {"code", "devenv"}
                )
            self.controller.set_scroll_sensitivity(
                float(self.shared_state.get("scroll_sensitivity", 700.0))
            )
            calibration = dict(self.shared_state.get("calibration", {}))
            self.controller.set_calibration(
                click_pinch_threshold=float(calibration.get("click_pinch_threshold", 0.055)),
                draw_pinch_threshold=float(calibration.get("draw_pinch_threshold", 0.05)),
                dead_zone_radius=float(calibration.get("dead_zone_radius", 0.035)),
                confirmation_window=int(calibration.get("confirmation_window", 5)),
                drag_activation_frames=int(calibration.get("drag_activation_frames", 7)),
                drag_release_confirmation_frames=int(
                    calibration.get("drag_release_confirmation_frames", 3)
                ),
                canvas_alpha=float(calibration.get("canvas_alpha", 0.32)),
            )
            self.controller.set_gesture_map(
                dict(self.shared_state.get("gesture_map", self.controller.gesture_map))
            )
            tracking_result = self.tracker.process_frame(frame)
            self.controller.handle(tracking_result, self.shared_state)
            if self.shared_state.pop("request_voice_command", False):
                self.trigger_voice_command()
            elapsed = time.perf_counter() - start_time
            anchor = (tracking_result.frame.shape[1] - 70, 70)
            self.projection.project(tracking_result.frame, anchor, elapsed)
            self.overlay_aircanvas(tracking_result.frame)
            cv2.imshow("Virtual Mouse AirCanvas", tracking_result.frame)
            if cv2.waitKey(1) & 0xFF == 27:
                self.shared_state["running"] = False
                break
        capture.release()
        cv2.destroyAllWindows()

    def run(self) -> None:
        self.voice_agent.start()
        vision_thread = threading.Thread(target=self.vision_loop, daemon=True)
        vision_thread.start()
        try:
            self.panel.run()
        finally:
            self.shared_state["running"] = False
            self.voice_agent.stop()
            vision_thread.join(timeout=1.5)


if __name__ == "__main__":
    VirtualMouseApplication().run()
