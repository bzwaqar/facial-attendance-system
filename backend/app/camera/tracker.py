"""Lightweight spatial and IoU-based multi-face tracker across video frames."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.ml.models import DetectedFace

logger = logging.getLogger(__name__)


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Compute Intersection-over-Union (IoU) overlap between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area = inter_width * inter_height

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union_area = areaA + areaB - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


@dataclass
class TrackedFace:
    """State and spatial trajectory of a persistently tracked face."""

    track_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    landmarks: Optional[List[List[float]]] = None
    frames_tracked: int = 1
    lost_frames: int = 0
    last_recognition_frame: int = -1

    @property
    def centroid(self) -> Tuple[float, float]:
        """Compute the centroid (x_center, y_center) of the bounding box."""
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])


class FaceTracker:
    """Lightweight multi-face association tracker using bounding box IoU overlap.
    
    Assigns persistent track IDs and prevents running expensive feature recognition on every frame.
    """

    def __init__(
        self,
        iou_threshold: Optional[float] = None,
        max_lost_frames: Optional[int] = None,
    ):
        """Initialize face tracker.
        
        Args:
            iou_threshold: Minimum IoU overlap to associate a detection with an existing track.
            max_lost_frames: Number of consecutive missing frames before retiring a track.
        """
        self.iou_threshold = (
            iou_threshold if iou_threshold is not None else settings.TRACKER_IOU_THRESHOLD
        )
        self.max_lost_frames = (
            max_lost_frames if max_lost_frames is not None else settings.TRACKER_MAX_LOST_FRAMES
        )

        self._active_tracks: Dict[int, TrackedFace] = {}
        self._next_track_id: int = 1
        self._frame_count: int = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def active_tracks_count(self) -> int:
        return len(self._active_tracks)

    def should_run_recognition(
        self,
        track: TrackedFace,
        interval: Optional[int] = None,
    ) -> bool:
        """Determine if a tracked face requires ArcFace embedding recognition on current frame.
        
        Args:
            track: Target TrackedFace instance.
            interval: Recognition frame interval (defaults to settings.RECOGNITION_INTERVAL_FRAMES).
            
        Returns:
            True if never recognized or if interval has elapsed since last recognition.
        """
        target_interval = interval or settings.RECOGNITION_INTERVAL_FRAMES
        if track.last_recognition_frame < 0:
            return True
        return (self._frame_count - track.last_recognition_frame) >= target_interval

    def mark_recognized(self, track_id: int) -> None:
        """Update last recognized frame index for the specified track."""
        if track_id in self._active_tracks:
            self._active_tracks[track_id].last_recognition_frame = self._frame_count

    def update(self, detected_faces: List[DetectedFace]) -> List[TrackedFace]:
        """Update active face tracks with newly detected faces from the current video frame.
        
        Args:
            detected_faces: List of DetectedFace objects from SCRFD.
            
        Returns:
            List of active TrackedFace instances present in the current frame.
        """
        self._frame_count += 1

        # Case 1: No previous active tracks -> spawn all detections as new tracks
        if not self._active_tracks:
            for det in detected_faces:
                track = TrackedFace(
                    track_id=self._next_track_id,
                    bbox=list(det.bbox),
                    confidence=float(det.confidence),
                    landmarks=det.landmarks,
                )
                self._active_tracks[self._next_track_id] = track
                self._next_track_id += 1
            return list(self._active_tracks.values())

        # Case 2: No faces detected in current frame -> age all active tracks
        if not detected_faces:
            retried_ids = []
            for tid, track in self._active_tracks.items():
                track.lost_frames += 1
                if track.lost_frames > self.max_lost_frames:
                    retried_ids.append(tid)

            for tid in retried_ids:
                del self._active_tracks[tid]
            return []

        # Case 3: Pairwise IoU cost matching between existing tracks and new detections
        track_ids = list(self._active_tracks.keys())
        num_tracks = len(track_ids)
        num_dets = len(detected_faces)

        iou_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
        for t_idx, tid in enumerate(track_ids):
            track_box = self._active_tracks[tid].bbox
            for d_idx, det in enumerate(detected_faces):
                iou_matrix[t_idx, d_idx] = compute_iou(track_box, det.bbox)

        matched_tracks = set()
        matched_dets = set()

        # Greedy bipartite matching from highest IoU down to threshold
        if num_tracks > 0 and num_dets > 0:
            flat_indices = np.argsort(-iou_matrix, axis=None)
            for flat_idx in flat_indices:
                t_idx, d_idx = np.unravel_index(flat_idx, (num_tracks, num_dets))
                if iou_matrix[t_idx, d_idx] < self.iou_threshold:
                    break
                if t_idx in matched_tracks or d_idx in matched_dets:
                    continue

                # Match found: update track
                tid = track_ids[t_idx]
                det = detected_faces[d_idx]
                track = self._active_tracks[tid]

                track.bbox = list(det.bbox)
                track.confidence = float(det.confidence)
                track.landmarks = det.landmarks
                track.frames_tracked += 1
                track.lost_frames = 0

                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)

        # Handle unmatched detections -> spawn new tracks
        for d_idx, det in enumerate(detected_faces):
            if d_idx not in matched_dets:
                new_track = TrackedFace(
                    track_id=self._next_track_id,
                    bbox=list(det.bbox),
                    confidence=float(det.confidence),
                    landmarks=det.landmarks,
                )
                self._active_tracks[self._next_track_id] = new_track
                self._next_track_id += 1

        # Handle unmatched tracks -> age and prune stale tracks
        stale_track_ids = []
        for t_idx, tid in enumerate(track_ids):
            if t_idx not in matched_tracks:
                track = self._active_tracks[tid]
                track.lost_frames += 1
                if track.lost_frames > self.max_lost_frames:
                    stale_track_ids.append(tid)

        for tid in stale_track_ids:
            del self._active_tracks[tid]

        # Return tracks that were actively detected in this frame
        current_frame_tracks = [
            track for track in self._active_tracks.values() if track.lost_frames == 0
        ]
        return current_frame_tracks

    def reset(self) -> None:
        """Clear all active tracks and reset frame counters."""
        self._active_tracks.clear()
        self._next_track_id = 1
        self._frame_count = 0
