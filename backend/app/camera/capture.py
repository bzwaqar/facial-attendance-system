"""Camera and video stream capture abstraction using OpenCV."""

import logging
import time
from typing import Generator, Optional, Tuple, Union

import cv2
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class CameraCapture:
    """Thread-safe OpenCV video capture wrapper supporting webcams and video files."""

    def __init__(
        self,
        source: Optional[Union[int, str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
    ):
        """Initialize camera capture source.
        
        Args:
            source: Camera device index (e.g. 0) or video file path. Defaults to settings.CAMERA_INDEX.
            width: Desired frame width resolution. Defaults to settings.CAMERA_WIDTH.
            height: Desired frame height resolution. Defaults to settings.CAMERA_HEIGHT.
            fps: Desired capture frame rate. Defaults to settings.CAMERA_FPS.
        """
        self.source = source if source is not None else settings.CAMERA_INDEX
        self.target_width = width or settings.CAMERA_WIDTH
        self.target_height = height or settings.CAMERA_HEIGHT
        self.target_fps = fps or settings.CAMERA_FPS
        self.frame_delay = 1.0 / max(1, self.target_fps)

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_running: bool = False
        self._last_frame_time: float = 0.0

    def open(self) -> bool:
        """Open the video capture stream.
        
        Returns:
            True if the camera was opened successfully, False otherwise.
        """
        if self._cap is not None and self._cap.isOpened():
            return True

        logger.info("Opening video capture source: %s", self.source)
        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            logger.error("Failed to open camera/video source: %s", self.source)
            return False

        # Configure hardware/driver capture properties if supported
        if self.target_width and self.target_height:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        if self.target_fps:
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self._cap.get(cv2.CAP_PROP_FPS))
        logger.info(
            "Camera opened successfully: %dx%d @ ~%d FPS (Source: %s)",
            actual_w,
            actual_h,
            actual_fps,
            self.source,
        )

        self._is_running = True
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the capture stream.
        
        Returns:
            Tuple of (success_flag, bgr_frame_array or None).
        """
        if self._cap is None or not self._cap.isOpened():
            if not self.open():
                return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("Failed to grab frame from video source: %s", self.source)
            return False, None

        # Rate limiter to respect target FPS
        now = time.perf_counter()
        elapsed = now - self._last_frame_time
        if elapsed < self.frame_delay:
            time.sleep(self.frame_delay - elapsed)
        self._last_frame_time = time.perf_counter()

        return True, frame

    def stream_frames(self) -> Generator[np.ndarray, None, None]:
        """Yield valid frames continuously until stopped or stream terminates."""
        if not self.open():
            return

        while self._is_running:
            success, frame = self.read_frame()
            if not success or frame is None:
                logger.info("End of video stream or camera read failure.")
                break
            yield frame

    def stop(self) -> None:
        """Signal capture to stop streaming."""
        self._is_running = False

    def release(self) -> None:
        """Release underlying OpenCV VideoCapture resources."""
        self.stop()
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as exc:
                logger.warning("Exception while releasing VideoCapture: %s", exc)
            finally:
                self._cap = None
        logger.info("Camera capture source %s released cleanly.", self.source)

    @property
    def is_opened(self) -> bool:
        """Return True if capture stream is currently active and open."""
        return self._cap is not None and self._cap.isOpened()

    def __enter__(self) -> "CameraCapture":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
