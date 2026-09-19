"""Concrete liveness detectors: TemporalHeuristic, ChallengeResponse, and Passthrough."""

from collections import deque
import logging
import time
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np

from app.core.config import settings
from app.liveness.base import (
    BaseLivenessDetector,
    ChallengeType,
    LivenessResult,
    LivenessStatus,
)
from app.liveness.cues import (
    HeadPoseCues,
    compute_patch_difference,
    estimate_head_pose_cues,
    extract_eye_patches,
    extract_texture_cues,
)
from app.liveness.evaluator import get_liveness_evaluator
from app.ml.models import DetectedFace

logger = logging.getLogger(__name__)


class TemporalObservation:
    """A single frame observation record stored in the temporal history buffer."""

    __slots__ = ("timestamp", "pose", "left_eye_patch", "right_eye_patch", "texture")

    def __init__(
        self,
        timestamp: float,
        pose: Optional[HeadPoseCues],
        left_eye_patch: Optional[np.ndarray],
        right_eye_patch: Optional[np.ndarray],
        texture: Dict[str, float],
    ):
        self.timestamp = timestamp
        self.pose = pose
        self.left_eye_patch = left_eye_patch
        self.right_eye_patch = right_eye_patch
        self.texture = texture


class TemporalHeuristicLivenessDetector(BaseLivenessDetector):
    """Lightweight practical anti-spoofing detector leveraging multi-frame temporal cues.
    
    Evaluates:
    1. Landmark Head Pose Dynamics: Variance and range of yaw/pitch over consecutive frames
       to detect static printed photos or motionless screens held still.
    2. Eye Patch Dynamics: Temporal appearance differences to detect blinks or natural micro-motion.
    3. Passive Texture & Glare: Blur and reflection cues.
    
    DISCLAIMER: Provides baseline protection against obvious presentation attacks in an
    experimental setup; not intended as ISO/IEC 30107-3 PAD certified production anti-spoofing.
    """

    def __init__(
        self,
        required_frames: Optional[int] = None,
        pose_variance_threshold: Optional[float] = None,
        min_score: Optional[float] = None,
        window_size: int = 15,
    ):
        """Initialize detector with customizable thresholds."""
        self.required_frames = required_frames or settings.LIVENESS_REQUIRED_FRAMES
        self.pose_variance_threshold = (
            pose_variance_threshold or settings.LIVENESS_POSE_VARIANCE_THRESHOLD
        )
        self.min_score = min_score or settings.LIVENESS_MIN_SCORE
        self.window_size = window_size

        # Buffer holding temporal observations keyed by track_id
        self._tracks: Dict[int, Deque[TemporalObservation]] = {}
        # Untracked fallback buffer (used when track_id is None)
        self._default_buffer: Deque[TemporalObservation] = deque(maxlen=self.window_size)
        self.evaluator = get_liveness_evaluator()

    def _get_buffer(self, track_id: Optional[int]) -> Deque[TemporalObservation]:
        """Retrieve or initialize the observation buffer for a given track."""
        if track_id is None:
            return self._default_buffer
        if track_id not in self._tracks:
            self._tracks[track_id] = deque(maxlen=self.window_size)
        return self._tracks[track_id]

    def reset(self, track_id: Optional[int] = None) -> None:
        """Reset temporal observation history."""
        if track_id is None:
            self._tracks.clear()
            self._default_buffer.clear()
        elif track_id in self._tracks:
            self._tracks[track_id].clear()

    def is_live(
        self,
        frame: np.ndarray,
        face: Optional[Union[DetectedFace, Any]] = None,
        bbox: Optional[List[float]] = None,
        landmarks: Optional[List[List[float]]] = None,
        track_id: Optional[int] = None,
        **kwargs: Any,
    ) -> LivenessResult:
        """Evaluate liveness across temporal frames for the given face track."""
        if frame is None or frame.size == 0:
            return LivenessResult(
                is_live=False,
                score=0.0,
                status=LivenessStatus.SPOOF,
                reason="Invalid or empty image frame provided.",
            )

        # 1. Resolve bounding box and landmarks from arguments or face object
        resolved_bbox = bbox
        resolved_landmarks = landmarks

        if face is not None:
            if resolved_bbox is None and hasattr(face, "bbox"):
                resolved_bbox = face.bbox
            if resolved_landmarks is None and hasattr(face, "landmarks"):
                resolved_landmarks = face.landmarks

        # 2. Extract instantaneous frame cues
        pose_cues = estimate_head_pose_cues(resolved_landmarks, frame_shape=frame.shape[:2])
        eye_patches = (
            extract_eye_patches(frame, resolved_landmarks)
            if resolved_landmarks
            else None
        )
        left_patch, right_patch = eye_patches if eye_patches else (None, None)

        # Texture extraction on cropped face
        texture_cues = {"laplacian_variance": 50.0, "specular_ratio": 0.0, "contrast": 30.0}
        if resolved_bbox and len(resolved_bbox) >= 4:
            x1, y1, x2, y2 = [int(round(c)) for c in resolved_bbox[:4]]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                face_crop = frame[y1:y2, x1:x2]
                texture_cues = extract_texture_cues(face_crop)

        # 3. Append observation to temporal buffer
        buf = self._get_buffer(track_id)
        buf.append(
            TemporalObservation(
                timestamp=time.perf_counter(),
                pose=pose_cues,
                left_eye_patch=left_patch,
                right_eye_patch=right_patch,
                texture=texture_cues,
            )
        )

        obs_count = len(buf)

        # 4. Check if sufficient temporal history is accumulated
        if obs_count < self.required_frames:
            result = LivenessResult(
                is_live=False,
                score=0.35,
                status=LivenessStatus.UNCERTAIN,
                reason=f"Accumulating temporal observations ({obs_count}/{self.required_frames} frames)...",
                cues={"observed_frames": obs_count, "required_frames": self.required_frames},
            )
            return result

        # 5. Analyze temporal cues across the observation sequence
        poses = [o.pose for o in buf if o.pose is not None]
        if len(poses) < (self.required_frames // 2 + 1):
            result = LivenessResult(
                is_live=False,
                score=0.20,
                status=LivenessStatus.UNCERTAIN,
                reason="Insufficient valid facial landmarks detected across temporal window.",
                cues={"valid_pose_frames": len(poses)},
            )
            return result

        # 5A. Head Pose Dynamics (Yaw, Pitch, Roll variation)
        yaw_values = [p.yaw_proxy for p in poses]
        pitch_values = [p.pitch_proxy for p in poses]
        roll_values = [p.roll_degrees for p in poses]

        yaw_std = float(np.std(yaw_values))
        pitch_std = float(np.std(pitch_values))
        roll_std = float(np.std(roll_values))

        yaw_range = float(np.ptp(yaw_values))
        pitch_range = float(np.ptp(pitch_values))

        # Composite pose motion score
        total_pose_variance = float(yaw_std + pitch_std + (roll_std / 30.0))

        # 5B. Eye patch frame-to-frame dynamics (Blink / micro-motion cue)
        eye_diffs: List[float] = []
        for i in range(1, len(buf)):
            prev_obs = buf[i - 1]
            curr_obs = buf[i]
            if (
                prev_obs.left_eye_patch is not None
                and curr_obs.left_eye_patch is not None
                and prev_obs.right_eye_patch is not None
                and curr_obs.right_eye_patch is not None
            ):
                diff_l = compute_patch_difference(prev_obs.left_eye_patch, curr_obs.left_eye_patch)
                diff_r = compute_patch_difference(prev_obs.right_eye_patch, curr_obs.right_eye_patch)
                eye_diffs.append((diff_l + diff_r) * 0.5)

        max_eye_diff = float(max(eye_diffs)) if eye_diffs else 0.0
        mean_eye_diff = float(np.mean(eye_diffs)) if eye_diffs else 0.0
        blink_detected = max_eye_diff >= settings.LIVENESS_BLINK_EAR_THRESHOLD

        # 5C. Passive Texture / Artifact Evaluation
        latest_texture = buf[-1].texture
        lap_var = latest_texture.get("laplacian_variance", 0.0)
        specular = latest_texture.get("specular_ratio", 0.0)

        texture_score = 1.0
        if lap_var < 10.0:  # Excessive blur (e.g. out-of-focus or low-quality printed copy)
            texture_score *= 0.5
        if specular > 0.30:  # Excessive specular glare (e.g. tablet/phone display reflection)
            texture_score *= 0.6

        # 6. Presentation Attack Detection (Static Photo Analysis)
        # Static photos held still have almost ZERO landmark pose variance across frames
        is_static_presentation = (
            total_pose_variance < self.pose_variance_threshold
            and not blink_detected
            and max_eye_diff < 0.05
        )

        # 7. Compute weighted composite confidence score
        # Motion variance is the dominant cue (60%), Eye dynamics (25%), Texture (15%)
        if total_pose_variance >= self.pose_variance_threshold:
            motion_subscore = float(np.clip(0.60 + 0.40 * (total_pose_variance / (self.pose_variance_threshold * 2.0)), 0.0, 1.0))
        else:
            motion_subscore = float(np.clip(total_pose_variance / (self.pose_variance_threshold * 2.0), 0.0, 0.45))
        eye_subscore = 1.0 if blink_detected else float(np.clip(max_eye_diff / 0.12, 0.2, 0.8))

        composite_score = float(
            (0.60 * motion_subscore) + (0.25 * eye_subscore) + (0.15 * texture_score)
        )
        composite_score = float(np.clip(composite_score, 0.0, 1.0))

        cues_telemetry = {
            "total_pose_variance": total_pose_variance,
            "yaw_std": yaw_std,
            "pitch_std": pitch_std,
            "roll_std": roll_std,
            "yaw_range": yaw_range,
            "pitch_range": pitch_range,
            "max_eye_diff": max_eye_diff,
            "mean_eye_diff": mean_eye_diff,
            "blink_detected": blink_detected,
            "laplacian_variance": lap_var,
            "specular_ratio": specular,
            "composite_score": composite_score,
            "observations_count": obs_count,
        }

        # 8. Decision Logic
        if is_static_presentation:
            status = LivenessStatus.SPOOF
            is_live_verdict = False
            reason = (
                f"Static presentation attack suspected: near-zero landmark variation "
                f"(variance: {total_pose_variance:.4f} < {self.pose_variance_threshold:.4f}, no blink)."
            )
            # Cap spoof score low
            final_score = min(composite_score, 0.25)

        elif composite_score >= self.min_score:
            status = LivenessStatus.LIVE
            is_live_verdict = True
            reasons = []
            if total_pose_variance >= self.pose_variance_threshold:
                reasons.append("natural head pose micro-dynamics verified")
            if blink_detected:
                reasons.append("natural eye blink detected")
            reason = f"Liveness confirmed: {', '.join(reasons) or 'organic temporal cues'}"
            final_score = composite_score

        else:
            status = LivenessStatus.UNCERTAIN
            is_live_verdict = False
            reason = (
                f"Liveness score ({composite_score:.3f}) fell below required threshold ({self.min_score:.3f})."
            )
            final_score = composite_score

        result = LivenessResult(
            is_live=is_live_verdict,
            score=final_score,
            status=status,
            reason=reason,
            cues=cues_telemetry,
        )

        # 9. Audit Logging
        self.evaluator.log_evaluation(
            result=result,
            track_id=track_id,
            extra_metadata=kwargs.get("metadata"),
        )

        return result


class ChallengeResponseLivenessDetector(BaseLivenessDetector):
    """Active challenge-response liveness detector.
    
    Prompts the subject to execute a specific voluntary physical action
    (e.g., turn head left, turn head right, nod, or blink) and confirms liveness
    only when the expected temporal cue sequence is verified within a timeout window.
    """

    def __init__(
        self,
        challenge_timeout_frames: int = 45,  # ~1.5 - 2 seconds at 30 fps
        fallback_detector: Optional[BaseLivenessDetector] = None,
    ):
        """Initialize challenge-response detector."""
        self.timeout_frames = challenge_timeout_frames
        self.fallback_detector = fallback_detector or TemporalHeuristicLivenessDetector(required_frames=3)
        self.evaluator = get_liveness_evaluator()

        # State tracking per track_id: (active_challenge, start_frame_count, initial_pose)
        self._challenge_states: Dict[int, Dict[str, Any]] = {}
        self._frame_counters: Dict[int, int] = {}

    def reset(self, track_id: Optional[int] = None) -> None:
        """Reset challenge-response states."""
        if track_id is None:
            self._challenge_states.clear()
            self._frame_counters.clear()
            self.fallback_detector.reset()
        else:
            self._challenge_states.pop(track_id, None)
            self._frame_counters.pop(track_id, None)
            self.fallback_detector.reset(track_id)

    def set_challenge(self, track_id: int, challenge: ChallengeType) -> None:
        """Explicitly assign an active challenge to a specific track."""
        self._challenge_states[track_id] = {
            "challenge": challenge,
            "start_frame": self._frame_counters.get(track_id, 0),
            "baseline_pose": None,
            "baseline_patch": None,
            "completed": False,
        }

    def is_live(
        self,
        frame: np.ndarray,
        face: Optional[Union[DetectedFace, Any]] = None,
        bbox: Optional[List[float]] = None,
        landmarks: Optional[List[List[float]]] = None,
        track_id: Optional[int] = None,
        **kwargs: Any,
    ) -> LivenessResult:
        """Evaluate whether the subject responded to the required challenge."""
        tid = track_id or 0
        self._frame_counters[tid] = self._frame_counters.get(tid, 0) + 1
        current_frame = self._frame_counters[tid]

        # Extract landmarks and pose
        resolved_landmarks = landmarks or (face.landmarks if face and hasattr(face, "landmarks") else None)
        pose = estimate_head_pose_cues(resolved_landmarks, frame_shape=frame.shape[:2]) if resolved_landmarks else None

        eye_patches = extract_eye_patches(frame, resolved_landmarks) if resolved_landmarks else None
        left_patch, right_patch = eye_patches if eye_patches else (None, None)

        # If no active challenge set for this track, assign default TURN_LEFT / ANY_MOVEMENT
        if tid not in self._challenge_states:
            chosen = kwargs.get("challenge", ChallengeType.TURN_LEFT)
            self._challenge_states[tid] = {
                "challenge": chosen,
                "start_frame": current_frame,
                "baseline_pose": pose,
                "baseline_patch": left_patch,
                "completed": False,
            }

        state = self._challenge_states[tid]
        challenge = state["challenge"]
        start_frame = state["start_frame"]
        frames_elapsed = current_frame - start_frame

        # Establish baseline pose if not yet saved
        if state["baseline_pose"] is None and pose is not None:
            state["baseline_pose"] = pose

        baseline_pose = state.get("baseline_pose")

        # 1. Check if challenge was already verified
        if state.get("completed"):
            return LivenessResult(
                is_live=True,
                score=0.95,
                status=LivenessStatus.LIVE,
                reason=f"Challenge '{challenge.value}' successfully verified.",
                active_challenge=challenge,
                cues={"challenge": challenge.value, "frames_elapsed": frames_elapsed},
            )

        # 2. Check for challenge timeout
        if frames_elapsed > self.timeout_frames:
            result = LivenessResult(
                is_live=False,
                score=0.15,
                status=LivenessStatus.SPOOF,
                reason=f"Challenge '{challenge.value}' timed out ({frames_elapsed} frames without required response).",
                active_challenge=challenge,
                cues={"challenge": challenge.value, "frames_elapsed": frames_elapsed},
            )
            self.evaluator.log_evaluation(result, track_id=tid)
            return result

        # 3. Evaluate challenge criteria
        success = False
        if pose is not None and baseline_pose is not None:
            if challenge == ChallengeType.TURN_LEFT:
                # Subject turns head left -> yaw_proxy shifts positively
                yaw_delta = pose.yaw_proxy - baseline_pose.yaw_proxy
                if pose.yaw_proxy > 0.18 or yaw_delta > 0.15:
                    success = True

            elif challenge == ChallengeType.TURN_RIGHT:
                # Subject turns head right -> yaw_proxy shifts negatively
                yaw_delta = pose.yaw_proxy - baseline_pose.yaw_proxy
                if pose.yaw_proxy < -0.18 or yaw_delta < -0.15:
                    success = True

            elif challenge == ChallengeType.NOD_HEAD:
                pitch_delta = abs(pose.pitch_proxy - baseline_pose.pitch_proxy)
                if pitch_delta > 0.18:
                    success = True

            elif challenge == ChallengeType.ANY_MOVEMENT:
                pose_delta = (
                    abs(pose.yaw_proxy - baseline_pose.yaw_proxy)
                    + abs(pose.pitch_proxy - baseline_pose.pitch_proxy)
                )
                if pose_delta > 0.12:
                    success = True

        if challenge == ChallengeType.BLINK:
            base_patch = state.get("baseline_patch")
            if base_patch is not None and left_patch is not None:
                diff = compute_patch_difference(base_patch, left_patch)
                if diff >= settings.LIVENESS_BLINK_EAR_THRESHOLD:
                    success = True

        if success:
            state["completed"] = True
            result = LivenessResult(
                is_live=True,
                score=0.95,
                status=LivenessStatus.LIVE,
                reason=f"Challenge '{challenge.value}' successfully executed.",
                active_challenge=challenge,
                cues={"challenge": challenge.value, "frames_elapsed": frames_elapsed},
            )
            self.evaluator.log_evaluation(result, track_id=tid)
            return result

        # Challenge still pending
        return LivenessResult(
            is_live=False,
            score=0.50,
            status=LivenessStatus.PENDING_CHALLENGE,
            reason=f"Please {challenge.value.replace('_', ' ').title()} to verify liveness ({frames_elapsed}/{self.timeout_frames})...",
            active_challenge=challenge,
            cues={"challenge": challenge.value, "frames_elapsed": frames_elapsed},
        )


class PassthroughLivenessDetector(BaseLivenessDetector):
    """Bypass detector that unconditionally marks presentations as LIVE (e.g. for testing/disabled state)."""

    def is_live(
        self,
        frame: np.ndarray,
        face: Optional[Union[DetectedFace, Any]] = None,
        bbox: Optional[List[float]] = None,
        landmarks: Optional[List[List[float]]] = None,
        track_id: Optional[int] = None,
        **kwargs: Any,
    ) -> LivenessResult:
        """Return constant LIVE result."""
        return LivenessResult(
            is_live=True,
            score=1.0,
            status=LivenessStatus.LIVE,
            reason="Liveness verification disabled / bypassed (Passthrough).",
            cues={"passthrough": True},
        )

    def reset(self, track_id: Optional[int] = None) -> None:
        """No-op reset for stateless passthrough."""
        pass
