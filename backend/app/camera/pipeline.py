"""Real-time camera recognition pipeline orchestrating detection, tracking, recognition, and verification."""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.camera.annotator import FrameAnnotator
from app.camera.temporal import TemporalFaceVerifier, TrackTemporalState
from app.camera.tracker import FaceTracker, TrackedFace, compute_iou
from app.core.config import settings
from app.liveness import BaseLivenessDetector, LivenessResult, get_liveness_detector
from app.ml.engine import FaceEngine, get_face_engine
from app.ml.models import DetectedFace
from app.services.matching_service import (
    IdentityMatchingService,
    get_identity_matching_service,
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessedFrameResult:
    """Consolidated outcome of processing a single video frame."""

    annotated_frame: np.ndarray
    tracks: List[TrackedFace] = field(default_factory=list)
    temporal_states: List[Optional[TrackTemporalState]] = field(default_factory=list)
    liveness_results: List[Optional[LivenessResult]] = field(default_factory=list)
    faces_detected_count: int = 0
    recognitions_executed_count: int = 0
    inference_time_ms: float = 0.0
    fps: float = 0.0


class CameraRecognitionPipeline:
    """Orchestrator integrating SCRFD detection, lightweight tracking, periodic ArcFace recognition,
    modular anti-spoofing liveness, and temporal verification into a unified real-time vision pipeline.
    """

    def __init__(
        self,
        face_engine: Optional[FaceEngine] = None,
        matching_service: Optional[IdentityMatchingService] = None,
        tracker: Optional[FaceTracker] = None,
        verifier: Optional[TemporalFaceVerifier] = None,
        annotator: Optional[FrameAnnotator] = None,
        liveness_detector: Optional[BaseLivenessDetector] = None,
        recognition_interval: Optional[int] = None,
        threshold: Optional[float] = None,
    ):
        """Initialize camera recognition pipeline with modular components."""
        self.face_engine = face_engine or get_face_engine()
        self.matching_service = matching_service or get_identity_matching_service()
        self.tracker = tracker or FaceTracker()
        self.verifier = verifier or TemporalFaceVerifier()
        self.annotator = annotator or FrameAnnotator()
        self.liveness_detector = liveness_detector or get_liveness_detector()

        self.recognition_interval = (
            recognition_interval
            if recognition_interval is not None
            else settings.RECOGNITION_INTERVAL_FRAMES
        )
        self.threshold = (
            threshold
            if threshold is not None
            else settings.FACE_SIMILARITY_THRESHOLD
        )

        self._student_names: Dict[str, str] = {}
        self._fps_history: List[float] = []

    def set_student_names(self, mapping: Dict[str, str]) -> None:
        """Set lookup mapping from student_id to full student display name."""
        self._student_names = dict(mapping)

    def process_frame(
        self,
        bgr_frame: np.ndarray,
        force_recognition: Optional[bool] = None,
    ) -> ProcessedFrameResult:
        """Process a single video frame through the complete recognition pipeline.
        
        1. Detect all faces via SCRFD (lightweight detection).
        2. Associate detections with active tracks using FaceTracker.
        3. For tracks requiring periodic recognition, extract ArcFace embeddings and query FAISS.
        4. Update temporal identity observation sequences.
        5. Render visual HUD and telemetry overlays.
        
        Args:
            bgr_frame: Input BGR image array.
            force_recognition: If True, forces ArcFace recognition on all tracks;
                               if False, suppresses recognition; if None, uses periodic interval.
                               
        Returns:
            ProcessedFrameResult containing annotated frame, track states, and metrics.
        """
        start_time = time.perf_counter()

        # Step 1: Detect all faces in frame (lightweight detection only)
        detected_faces = self.face_engine.detect_faces_only(
            bgr_frame,
            confidence_threshold=settings.DETECTION_CONFIDENCE_THRESHOLD,
        )

        # Step 2: Multi-face spatial tracking
        active_tracks = self.tracker.update(detected_faces)

        # Match active tracks to detected face objects by IoU to access landmarks
        track_to_det: Dict[int, DetectedFace] = {}
        for track in active_tracks:
            best_iou = 0.0
            best_det: Optional[DetectedFace] = None
            for det in detected_faces:
                iou = compute_iou(track.bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det
            if best_det and best_iou > 0.4:
                track_to_det[track.track_id] = best_det

        # Step 3 & 4: Liveness verification, periodic recognition, and temporal updates
        recognitions_executed = 0
        states: List[Optional[TrackTemporalState]] = []
        liveness_results: List[Optional[LivenessResult]] = []

        for track in active_tracks:
            det_face = track_to_det.get(track.track_id)

            # Step 3A: Anti-spoofing / liveness evaluation
            liveness_res: Optional[LivenessResult] = None
            if self.liveness_detector is not None:
                try:
                    liveness_res = self.liveness_detector.is_live(
                        frame=bgr_frame,
                        face=det_face,
                        bbox=track.bbox,
                        landmarks=track.landmarks,
                        track_id=track.track_id,
                    )
                    self.verifier.update_liveness(
                        track_id=track.track_id,
                        is_live=liveness_res.is_live,
                        score=liveness_res.score,
                        status=liveness_res.status.value,
                        reason=liveness_res.reason,
                    )
                except Exception as exc:
                    logger.warning("Error during liveness evaluation for track #%d: %s", track.track_id, exc)

            liveness_results.append(liveness_res)

            # Step 3B: Determine if recognition should run on this frame
            if force_recognition is True:
                run_rec = True
            elif force_recognition is False:
                run_rec = False
            else:
                run_rec = self.tracker.should_run_recognition(track, self.recognition_interval)

            if run_rec and det_face is not None:
                try:
                    # Run ArcFace feature extraction ONLY when periodic interval fires
                    emb = self.face_engine.extract_face_embedding(bgr_frame, det_face)

                    # Query FAISS vector index
                    match_result = self.matching_service.match_face_embedding(
                        query_vector=emb,
                        threshold=self.threshold,
                        top_k=3,
                    )

                    student_name = (
                        self._student_names.get(match_result.student_id)
                        if match_result.student_id
                        else None
                    )

                    # Update temporal observation sequence
                    self.verifier.add_observation(
                        track_id=track.track_id,
                        match_result=match_result,
                        student_name=student_name,
                    )
                    self.tracker.mark_recognized(track.track_id)
                    recognitions_executed += 1

                except Exception as exc:
                    logger.warning("Error during periodic recognition for track #%d: %s", track.track_id, exc)

            state = self.verifier.get_track_state(track.track_id)
            states.append(state)

        # Clean up stale track histories in temporal verifier and liveness detector
        active_ids = [t.track_id for t in active_tracks]
        self.verifier.cleanup_stale_tracks(active_ids)
        if self.liveness_detector is not None:
            # Clean up internal tracks if supported
            tracks_dict = getattr(self.liveness_detector, "_tracks", None)
            if isinstance(tracks_dict, dict):
                stale_ids = [tid for tid in list(tracks_dict.keys()) if tid not in active_ids]
                for tid in stale_ids:
                    self.liveness_detector.reset(tid)

        # Step 5: Metrics and FPS calculation
        elapsed_sec = max(1e-5, time.perf_counter() - start_time)
        instant_fps = 1.0 / elapsed_sec
        self._fps_history.append(instant_fps)
        if len(self._fps_history) > 20:
            self._fps_history.pop(0)
        smooth_fps = float(np.mean(self._fps_history))

        # Step 6: Render HUD overlay
        annotated = self.annotator.annotate_frame(
            frame=bgr_frame,
            tracks=active_tracks,
            states=states,
            fps=smooth_fps,
        )

        return ProcessedFrameResult(
            annotated_frame=annotated,
            tracks=active_tracks,
            temporal_states=states,
            liveness_results=liveness_results,
            faces_detected_count=len(detected_faces),
            recognitions_executed_count=recognitions_executed,
            inference_time_ms=round(elapsed_sec * 1000.0, 2),
            fps=round(smooth_fps, 1),
        )
