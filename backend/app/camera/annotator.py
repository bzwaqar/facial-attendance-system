"""Frame visualizer and HUD annotation overlay for real-time camera recognition."""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.camera.temporal import TemporalIdentityState, TrackTemporalState
from app.camera.tracker import TrackedFace


class FrameAnnotator:
    """Renders styled HUD bounding boxes, identity tags, and status statistics on video frames."""

    # Color Palette in BGR format
    COLOR_CONFIRMED = (46, 204, 113)  # Bright Emerald Green
    COLOR_UNKNOWN = (60, 60, 235)     # Deep Crimson Red
    COLOR_VERIFYING = (0, 165, 255)   # Amber Orange
    COLOR_SPOOF = (0, 0, 255)         # Neon Bright Red (Presentation Attack Alert)
    COLOR_CHALLENGE = (0, 215, 255)   # Bright Gold / Amber (Interactive Challenge)
    COLOR_CORNER = (255, 255, 255)    # White accent corners

    def __init__(self):
        """Initialize frame annotator."""
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def get_state_color(
        self,
        state: TemporalIdentityState,
        temporal_state: Optional[TrackTemporalState] = None,
    ) -> Tuple[int, int, int]:
        """Map temporal verification and anti-spoofing state to display color."""
        if temporal_state is not None:
            if temporal_state.is_spoof:
                return self.COLOR_SPOOF
            if temporal_state.liveness_status == "PENDING_CHALLENGE":
                return self.COLOR_CHALLENGE

        if state == TemporalIdentityState.CONFIRMED:
            return self.COLOR_CONFIRMED
        elif state == TemporalIdentityState.UNKNOWN:
            return self.COLOR_UNKNOWN
        else:
            return self.COLOR_VERIFYING

    def draw_corner_brackets(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        corner_len: int = 15,
        thickness: int = 2,
    ) -> None:
        """Draw modern corner bracket accents on bounding box corners."""
        x1, y1 = pt1
        x2, y2 = pt2

        # Top-Left corner
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)

        # Top-Right corner
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)

        # Bottom-Left corner
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)

        # Bottom-Right corner
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)

    def annotate_face(
        self,
        frame: np.ndarray,
        track: TrackedFace,
        temporal_state: Optional[TrackTemporalState],
    ) -> None:
        """Draw styled bounding box and identity information for a tracked face.
        
        Args:
            frame: Target BGR image array (modified in place).
            track: TrackedFace instance.
            temporal_state: Current temporal verification state for this track.
        """
        x1, y1, x2, y2 = [int(round(c)) for c in track.bbox]
        state = temporal_state.state if temporal_state else TemporalIdentityState.VERIFYING
        color = self.get_state_color(state, temporal_state)

        # 1. Main bounding rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        # 2. Sleek corner brackets
        self.draw_corner_brackets(frame, (x1, y1), (x2, y2), color=color, corner_len=18, thickness=2)

        # 3. Label text formatting
        if temporal_state:
            label = temporal_state.display_label
        else:
            label = f"Track #{track.track_id} (Tracking...)"

        # Text size calculation
        font_scale = 0.52
        font_thick = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, self.font, font_scale, font_thick)

        # Draw label background pill above bounding box
        label_y1 = max(0, y1 - text_h - 10)
        label_y2 = y1
        label_x2 = min(frame.shape[1], x1 + text_w + 14)

        cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
        # Text in contrasting black or white
        text_color = (0, 0, 0) if (state != TemporalIdentityState.UNKNOWN and not (temporal_state and temporal_state.is_spoof)) else (255, 255, 255)
        cv2.putText(
            frame,
            label,
            (x1 + 6, y1 - 6),
            self.font,
            font_scale,
            text_color,
            font_thick,
            cv2.LINE_AA,
        )

        # Optional: draw landmarks if available
        if track.landmarks:
            for pt in track.landmarks:
                cv2.circle(frame, (int(round(pt[0])), int(round(pt[1]))), 2, (0, 255, 255), -1, cv2.LINE_AA)

    def draw_top_hud(
        self,
        frame: np.ndarray,
        fps: float,
        active_faces: int,
        confirmed_count: int,
        spoof_count: int = 0,
    ) -> None:
        """Draw top telemetry banner with FPS, active tracks, confirmed counts, and spoof alerts."""
        h, w = frame.shape[:2]
        hud_h = 32

        # Translucent dark banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Telemetry string
        spoof_str = f" | Spoofs: {spoof_count}" if spoof_count > 0 else ""
        hud_text = (
            f"FPS: {fps:4.1f} | Active Tracks: {active_faces} | "
            f"Confirmed: {confirmed_count}{spoof_str} | Press 'q' to Quit"
        )
        # Display alert in red if active spoof attack detected
        hud_color = (0, 0, 255) if spoof_count > 0 else (230, 230, 230)
        cv2.putText(
            frame,
            hud_text,
            (12, 21),
            self.font,
            0.50,
            hud_color,
            1,
            cv2.LINE_AA,
        )

    def annotate_frame(
        self,
        frame: np.ndarray,
        tracks: List[TrackedFace],
        states: List[Optional[TrackTemporalState]],
        fps: float = 0.0,
    ) -> np.ndarray:
        """Annotate entire video frame with all active face tracks and telemetry.
        
        Args:
            frame: Raw BGR input frame.
            tracks: List of active TrackedFace objects.
            states: Parallel list of temporal states for each track.
            fps: Current real-time processing FPS.
            
        Returns:
            Annotated BGR frame copy.
        """
        annotated = frame.copy()
        confirmed_count = sum(
            1 for s in states if s and s.state == TemporalIdentityState.CONFIRMED
        )
        spoof_count = sum(
            1 for s in states if s and s.is_spoof
        )

        for track, state in zip(tracks, states):
            self.annotate_face(annotated, track, state)

        self.draw_top_hud(
            frame=annotated,
            fps=fps,
            active_faces=len(tracks),
            confirmed_count=confirmed_count,
            spoof_count=spoof_count,
        )
        return annotated
