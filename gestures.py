from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pygetwindow as gw
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from filters import SignalProcessor


@dataclass
class TrackingResult:
    frame: np.ndarray
    landmarks: list[tuple[float, float]] = field(default_factory=list)
    handedness: str = ""
    pinch_distance: float = 1.0
    middle_pinch_distance: float = 1.0
    drawing_pinch_distance: float = 1.0
    has_hand: bool = False


class HandTracker:
    def __init__(
        self,
        frame_width: int = 600,
        frame_height: int = 400,
        temporal_alpha: float = 0.45,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.temporal_alpha = temporal_alpha
        self.previous_landmarks: list[tuple[float, float]] = []
        self.video_timestamp_ms = 0
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
            num_hands=1,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

    def stabilize_landmarks(
        self, landmarks: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        if not self.previous_landmarks:
            self.previous_landmarks = landmarks
            return landmarks
        stabilized: list[tuple[float, float]] = []
        for current, previous in zip(landmarks, self.previous_landmarks):
            x_value = (
                self.temporal_alpha * current[0]
                + (1.0 - self.temporal_alpha) * previous[0]
            )
            y_value = (
                self.temporal_alpha * current[1]
                + (1.0 - self.temporal_alpha) * previous[1]
            )
            stabilized.append((x_value, y_value))
        self.previous_landmarks = stabilized
        return stabilized

    def process_frame(self, frame: np.ndarray) -> TrackingResult:
        resized = cv2.resize(frame, (self.frame_width, self.frame_height))
        resized = cv2.flip(resized, 1)
        processed = self.enhance_frame(resized)
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.video_timestamp_ms += 33
        result = self.hand_landmarker.detect_for_video(
            mp_image, self.video_timestamp_ms
        )
        tracking_result = TrackingResult(frame=resized)
        if not result.hand_landmarks:
            self.previous_landmarks = []
            return tracking_result
        hand_landmarks = result.hand_landmarks[0]
        handedness = ""
        if result.handedness:
            handedness = result.handedness[0][0].category_name
        normalized_landmarks = [
            (landmark.x, landmark.y) for landmark in hand_landmarks
        ]
        stabilized_landmarks = self.stabilize_landmarks(normalized_landmarks)
        self.draw_landmarks(resized, stabilized_landmarks)
        tracking_result.landmarks = stabilized_landmarks
        tracking_result.handedness = handedness
        tracking_result.has_hand = True
        tracking_result.pinch_distance = self.distance(stabilized_landmarks[4], stabilized_landmarks[8])
        tracking_result.middle_pinch_distance = self.distance(
            stabilized_landmarks[4], stabilized_landmarks[12]
        )
        tracking_result.drawing_pinch_distance = self.distance(
            stabilized_landmarks[4], stabilized_landmarks[16]
        )
        return tracking_result

    @staticmethod
    def enhance_frame(frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        merged = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return cv2.GaussianBlur(enhanced, (3, 3), 0)

    @staticmethod
    def distance(
        first_point: tuple[float, float], second_point: tuple[float, float]
    ) -> float:
        return float(math.dist(first_point, second_point))

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: list[tuple[float, float]],
    ) -> None:
        connections = (
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        )
        frame_height, frame_width = frame.shape[:2]
        pixel_points = [
            (
                int(point[0] * frame_width),
                int(point[1] * frame_height),
            )
            for point in landmarks
        ]
        for start_index, end_index in connections:
            cv2.line(frame, pixel_points[start_index], pixel_points[end_index], (0, 255, 0), 1)
        for point in pixel_points:
            cv2.circle(frame, point, 2, (255, 255, 255), -1)


class ProjectionEngine3D:
    def __init__(self, scale: float = 56.0) -> None:
        self.scale = scale
        self.vertices = np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.edges = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )

    def rotation_matrix(self, angle_x: float, angle_y: float) -> np.ndarray:
        sin_x = math.sin(angle_x)
        cos_x = math.cos(angle_x)
        sin_y = math.sin(angle_y)
        cos_y = math.cos(angle_y)
        rotation_x = np.array(
            [[1.0, 0.0, 0.0], [0.0, cos_x, -sin_x], [0.0, sin_x, cos_x]],
            dtype=np.float32,
        )
        rotation_y = np.array(
            [[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]],
            dtype=np.float32,
        )
        return rotation_y @ rotation_x

    def project(self, frame: np.ndarray, anchor: tuple[int, int], tick: float) -> None:
        height, width = frame.shape[:2]
        matrix = self.rotation_matrix(tick * 0.7, tick * 1.1)
        rotated = self.vertices @ matrix.T
        points: list[tuple[int, int]] = []
        for x_value, y_value, z_value in rotated:
            depth = z_value + 4.0
            projection_scale = self.scale / depth
            point_x = int(anchor[0] + x_value * projection_scale)
            point_y = int(anchor[1] + y_value * projection_scale)
            point_x = max(0, min(width - 1, point_x))
            point_y = max(0, min(height - 1, point_y))
            points.append((point_x, point_y))
        for start_index, end_index in self.edges:
            cv2.line(frame, points[start_index], points[end_index], (0, 255, 255), 1)


class GestureController:
    def __init__(
        self,
        scroll_sensitivity: float = 700.0,
        gesture_map: dict[str, str] | None = None,
    ) -> None:
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.0
        self.screen_width, self.screen_height = pyautogui.size()
        self.cursor_filter = SignalProcessor(
            process_noise=2e-4,
            measurement_noise=3e-3,
        )
        self.scroll_filter = SignalProcessor(
            process_noise=5e-4,
            measurement_noise=8e-3,
        )
        self.scroll_sensitivity = scroll_sensitivity
        self.profile_map = {
            "chrome": 1.45,
            "msedge": 1.45,
            "firefox": 1.35,
            "code": 0.45,
            "devenv": 0.5,
            "notepad": 0.85,
            "default": 1.0,
        }
        self.gesture_map = gesture_map or {
            "left_click": "thumb_index_pinch",
            "right_click": "thumb_middle_pinch",
            "draw_toggle": "thumb_ring_pinch",
        }
        self.last_left_click = 0.0
        self.last_right_click = 0.0
        self.last_voice_trigger = 0.0
        self.last_scroll_y: float | None = None
        self.aircanvas_enabled = False
        self.aircanvas_points: list[tuple[int, int] | None] = []
        self.drag_active = False
        self.drag_release_frames = 0
        self.index_pinch_frames = 0
        self.pending_click = False
        self.confirmation_window = 5
        self.gesture_buffer: deque[str] = deque(maxlen=self.confirmation_window)
        self.dead_zone_radius = 0.035
        self.dead_zone_reference: tuple[float, float] | None = None
        self.canvas_alpha = 0.32
        self.last_canvas_point: tuple[float, float] | None = None
        self.click_pinch_threshold = 0.055
        self.draw_pinch_threshold = 0.05
        self.drag_activation_frames = self.confirmation_window + 2
        self.drag_release_confirmation_frames = 3

    def set_scroll_sensitivity(self, sensitivity: float) -> None:
        self.scroll_sensitivity = sensitivity

    def set_gesture_map(self, gesture_map: dict[str, str]) -> None:
        self.gesture_map = gesture_map

    def set_calibration(
        self,
        click_pinch_threshold: float,
        draw_pinch_threshold: float,
        dead_zone_radius: float,
        confirmation_window: int,
        drag_activation_frames: int,
        drag_release_confirmation_frames: int,
        canvas_alpha: float,
    ) -> None:
        self.click_pinch_threshold = click_pinch_threshold
        self.draw_pinch_threshold = draw_pinch_threshold
        self.dead_zone_radius = dead_zone_radius
        self.confirmation_window = confirmation_window
        self.drag_activation_frames = drag_activation_frames
        self.drag_release_confirmation_frames = drag_release_confirmation_frames
        self.canvas_alpha = canvas_alpha
        if self.gesture_buffer.maxlen != self.confirmation_window:
            self.gesture_buffer = deque(self.gesture_buffer, maxlen=self.confirmation_window)

    def get_profile_multiplier(self, profile_name: str) -> float:
        return self.profile_map.get(profile_name, self.profile_map["default"])

    def set_cursor_precision(self, precise_mode: bool) -> None:
        if precise_mode:
            self.cursor_filter.set_process_noise(7e-5)
            self.scroll_filter.set_process_noise(2e-4)
        else:
            self.cursor_filter.set_process_noise(2e-4)
            self.scroll_filter.set_process_noise(5e-4)

    def update_gesture_buffer(self, candidate: str) -> str:
        self.gesture_buffer.append(candidate)
        if len(self.gesture_buffer) < self.confirmation_window:
            return "idle"
        if len(set(self.gesture_buffer)) == 1:
            return self.gesture_buffer[0]
        return "idle"

    def apply_dead_zone(self, point: tuple[float, float]) -> tuple[float, float]:
        center = (0.5, 0.5)
        if self.dead_zone_reference is None:
            self.dead_zone_reference = point
        distance_to_center = math.dist(point, center)
        distance_to_reference = math.dist(point, self.dead_zone_reference)
        if (
            distance_to_center <= self.dead_zone_radius
            and distance_to_reference <= self.dead_zone_radius * 0.75
        ):
            return self.dead_zone_reference
        self.dead_zone_reference = point
        return point

    def map_to_screen(
        self, point: tuple[float, float], frame_shape: tuple[int, int, int]
    ) -> tuple[int, int]:
        frame_height, frame_width = frame_shape[:2]
        stabilized_point = self.apply_dead_zone(point)
        x_value = stabilized_point[0] * frame_width
        y_value = stabilized_point[1] * frame_height
        normalized_x = min(max(x_value / frame_width, 0.0), 1.0)
        normalized_y = min(max(y_value / frame_height, 0.0), 1.0)
        filtered_x, filtered_y = self.cursor_filter.process(normalized_x, normalized_y)
        screen_x = int(min(max(filtered_x, 0.0), 1.0) * self.screen_width)
        screen_y = int(min(max(filtered_y, 0.0), 1.0) * self.screen_height)
        return screen_x, screen_y

    def perform_scroll(
        self,
        index_point: tuple[float, float],
        profile_multiplier: float,
    ) -> int:
        _, filtered_y = self.scroll_filter.process(index_point[0], index_point[1])
        if self.last_scroll_y is None:
            self.last_scroll_y = filtered_y
            return 0
        delta = self.last_scroll_y - filtered_y
        self.last_scroll_y = filtered_y
        scroll_amount = int(delta * self.scroll_sensitivity * profile_multiplier)
        if scroll_amount != 0:
            pyautogui.scroll(scroll_amount)
        return scroll_amount

    def update_aircanvas(
        self, point: tuple[int, int], drawing_active: bool
    ) -> list[tuple[int, int] | None]:
        if drawing_active:
            smoothed_point = self.smooth_canvas_point(point)
            self.aircanvas_points.append(smoothed_point)
        elif self.aircanvas_points and self.aircanvas_points[-1] is not None:
            self.aircanvas_points.append(None)
            self.last_canvas_point = None
        if len(self.aircanvas_points) > 2048:
            self.aircanvas_points = self.aircanvas_points[-2048:]
        return list(self.aircanvas_points)

    def smooth_canvas_point(self, point: tuple[int, int]) -> tuple[int, int]:
        point_x = float(point[0])
        point_y = float(point[1])
        if self.last_canvas_point is None:
            self.last_canvas_point = (point_x, point_y)
            return point
        smoothed_x = self.canvas_alpha * point_x + (1.0 - self.canvas_alpha) * self.last_canvas_point[0]
        smoothed_y = self.canvas_alpha * point_y + (1.0 - self.canvas_alpha) * self.last_canvas_point[1]
        self.last_canvas_point = (smoothed_x, smoothed_y)
        return int(smoothed_x), int(smoothed_y)

    def clear_canvas(self) -> None:
        self.aircanvas_points.clear()
        self.last_canvas_point = None

    def resolve_candidate(
        self,
        left_pinch: bool,
        right_pinch: bool,
        draw_pinch: bool,
        extended_fingers: int,
        voice_pose: bool,
    ) -> str:
        if draw_pinch and not left_pinch:
            return "draw_toggle"
        if voice_pose:
            return "voice_trigger"
        if extended_fingers >= 4:
            return "scroll"
        if right_pinch:
            return "right_click"
        if left_pinch:
            return "left_pinch"
        return "move"

    def handle_drag(
        self,
        left_pinch: bool,
        confirmed_gesture: str,
    ) -> str:
        if left_pinch:
            self.index_pinch_frames += 1
            self.drag_release_frames = 0
            if confirmed_gesture == "left_pinch":
                self.pending_click = True
            if self.pending_click and self.index_pinch_frames >= self.drag_activation_frames:
                if not self.drag_active:
                    pyautogui.mouseDown()
                    self.drag_active = True
                return "drag"
            return "hold"
        if self.drag_active:
            self.drag_release_frames += 1
            if self.drag_release_frames >= self.drag_release_confirmation_frames:
                pyautogui.mouseUp()
                self.drag_active = False
                self.drag_release_frames = 0
                self.pending_click = False
                self.index_pinch_frames = 0
                return "drag_release"
            return "drag"
        if self.pending_click and 0 < self.index_pinch_frames < self.drag_activation_frames:
            pyautogui.click()
            self.pending_click = False
            self.index_pinch_frames = 0
            return "left_click"
        self.pending_click = False
        self.index_pinch_frames = 0
        return "move"

    def handle(
        self, tracking_result: TrackingResult, shared_state: dict[str, Any]
    ) -> None:
        if shared_state.get("clear_canvas", False):
            self.clear_canvas()
            shared_state["aircanvas_points"] = []
            shared_state["clear_canvas"] = False
        profile_name = str(shared_state.get("active_profile", "default"))
        profile_multiplier = self.get_profile_multiplier(profile_name)
        shared_state["active_profile"] = profile_name
        if not tracking_result.has_hand:
            self.last_scroll_y = None
            if self.drag_active:
                pyautogui.mouseUp()
                self.drag_active = False
            self.pending_click = False
            self.index_pinch_frames = 0
            self.drag_release_frames = 0
            self.gesture_buffer.clear()
            self.dead_zone_reference = None
            shared_state["gesture"] = "idle"
            return
        index_point = tracking_result.landmarks[8]
        screen_x, screen_y = self.map_to_screen(index_point, tracking_result.frame.shape)
        pyautogui.moveTo(screen_x, screen_y)
        now = time.time()
        gesture_name = "move"
        gesture_inputs = {
            "thumb_index_pinch": tracking_result.pinch_distance < self.click_pinch_threshold,
            "thumb_middle_pinch": tracking_result.middle_pinch_distance < self.click_pinch_threshold,
            "thumb_ring_pinch": tracking_result.drawing_pinch_distance < self.draw_pinch_threshold,
        }
        left_pinch = gesture_inputs.get(
            self.gesture_map.get("left_click", "thumb_index_pinch"),
            False,
        )
        right_pinch = gesture_inputs.get(
            self.gesture_map.get("right_click", "thumb_middle_pinch"),
            False,
        )
        draw_pinch = gesture_inputs.get(
            self.gesture_map.get("draw_toggle", "thumb_ring_pinch"),
            False,
        )
        extended_fingers = self.count_extended_fingers(tracking_result.landmarks)
        voice_pose = self.is_voice_pose(tracking_result.landmarks, left_pinch, right_pinch, draw_pinch)
        candidate = self.resolve_candidate(
            left_pinch,
            right_pinch,
            draw_pinch,
            extended_fingers,
            voice_pose,
        )
        confirmed_gesture = self.update_gesture_buffer(candidate)
        if confirmed_gesture == "voice_trigger" and now - self.last_voice_trigger > 2.0:
            shared_state["request_voice_command"] = True
            shared_state["system_status"] = "Voice command listening"
            self.last_voice_trigger = now
            gesture_name = "voice_trigger"
        elif confirmed_gesture == "right_click" and now - self.last_right_click > 0.45:
            pyautogui.rightClick()
            self.last_right_click = now
            gesture_name = "right_click"
        elif confirmed_gesture == "scroll":
            scroll_amount = self.perform_scroll(index_point, profile_multiplier)
            if scroll_amount:
                gesture_name = "scroll"
        else:
            gesture_name = self.handle_drag(
                left_pinch,
                confirmed_gesture,
            )
        if candidate != "scroll":
            self.last_scroll_y = None
        if confirmed_gesture == "draw_toggle" and not self.aircanvas_enabled:
            self.aircanvas_enabled = True
        elif confirmed_gesture == "move" and self.aircanvas_enabled and not left_pinch:
            self.aircanvas_enabled = False
        drawing_active = self.aircanvas_enabled and (left_pinch or self.drag_active)
        points = self.update_aircanvas((screen_x, screen_y), drawing_active)
        shared_state["aircanvas_points"] = points
        shared_state["aircanvas_enabled"] = self.aircanvas_enabled
        shared_state["cursor"] = (screen_x, screen_y)
        shared_state["gesture"] = gesture_name
        shared_state["pinch_distance"] = tracking_result.pinch_distance
        if gesture_name == "left_click":
            self.last_left_click = now
        if gesture_name in {"drag", "drag_release"}:
            shared_state["system_status"] = gesture_name

    @staticmethod
    def count_extended_fingers(landmarks: list[tuple[float, float]]) -> int:
        finger_pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
        return sum(1 for tip, pip in finger_pairs if landmarks[tip][1] < landmarks[pip][1])

    @staticmethod
    def is_voice_pose(
        landmarks: list[tuple[float, float]],
        left_pinch: bool,
        right_pinch: bool,
        draw_pinch: bool,
    ) -> bool:
        if left_pinch or right_pinch or draw_pinch:
            return False
        index_up = landmarks[8][1] < landmarks[6][1]
        middle_up = landmarks[12][1] < landmarks[10][1]
        ring_down = landmarks[16][1] >= landmarks[14][1]
        pinky_down = landmarks[20][1] >= landmarks[18][1]
        return index_up and middle_up and ring_down and pinky_down


def detect_active_profile() -> str:
    try:
        active_window = gw.getActiveWindow()
        title = active_window.title.lower() if active_window and active_window.title else ""
    except Exception:
        title = ""
    if "chrome" in title:
        return "chrome"
    if "edge" in title:
        return "msedge"
    if "firefox" in title:
        return "firefox"
    if "visual studio code" in title or "code" in title:
        return "code"
    if "visual studio" in title:
        return "devenv"
    if "notepad" in title:
        return "notepad"
    return "default"
