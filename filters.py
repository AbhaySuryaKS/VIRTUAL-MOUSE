from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SignalProcessor:
    process_noise: float = 1e-4
    measurement_noise: float = 1e-2
    state: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 1), dtype=np.float32)
    )
    covariance: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float32)
    )

    def __post_init__(self) -> None:
        self.transition = np.array(
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0],
             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        self.measurement = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        self.process_covariance = np.eye(4, dtype=np.float32) * self.process_noise
        self.measurement_covariance = (
            np.eye(2, dtype=np.float32) * self.measurement_noise
        )
        self.identity = np.eye(4, dtype=np.float32)
        self.initialized = False

    def reset(self) -> None:
        self.state[:] = 0.0
        self.covariance[:] = self.identity
        self.initialized = False

    def set_process_noise(self, process_noise: float) -> None:
        self.process_noise = process_noise
        self.process_covariance = np.eye(4, dtype=np.float32) * self.process_noise

    def process(self, x: float, y: float) -> tuple[float, float]:
        measurement = np.array([[x], [y]], dtype=np.float32)
        if not self.initialized:
            self.state[0, 0] = x
            self.state[1, 0] = y
            self.initialized = True
            return x, y
        self.state = self.transition @ self.state
        self.covariance = (
            self.transition @ self.covariance @ self.transition.T
            + self.process_covariance
        )
        innovation = measurement - (self.measurement @ self.state)
        innovation_covariance = (
            self.measurement @ self.covariance @ self.measurement.T
            + self.measurement_covariance
        )
        kalman_gain = (
            self.covariance
            @ self.measurement.T
            @ np.linalg.inv(innovation_covariance)
        )
        self.state = self.state + kalman_gain @ innovation
        self.covariance = (
            self.identity - kalman_gain @ self.measurement
        ) @ self.covariance
        return float(self.state[0, 0]), float(self.state[1, 0])
