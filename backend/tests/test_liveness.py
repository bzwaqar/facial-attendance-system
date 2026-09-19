"""Unit tests for anti-spoofing and liveness detection module."""

import numpy as np
import pytest

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
from app.liveness.detector import (
    ChallengeResponseLivenessDetector,
    PassthroughLivenessDetector,
    TemporalHeuristicLivenessDetector,
)
from app.liveness.evaluator import LivenessEvaluator
from app.liveness.factory import get_liveness_detector


def _make_dummy_frame_and_landmarks(yaw_shift: float = 0.0, pitch_shift: float = 0.0):
    """Helper to generate a dummy frame and SCRFD-style 5 landmarks."""
    frame = np.ones((200, 200, 3), dtype=np.uint8) * 120
    cx, cy = 100, 100
    le = [float(cx - 30), float(cy - 20)]
    re = [float(cx + 30), float(cy - 20)]
    nose = [float(cx + yaw_shift * 30), float(cy + pitch_shift * 25)]
    lm = [float(cx - 20), float(cy + 25)]
    rm = [float(cx + 20), float(cy + 25)]
    landmarks = [le, re, nose, lm, rm]
    bbox = [float(cx - 50), float(cy - 50), float(cx + 50), float(cy + 50)]
    return frame, landmarks, bbox


def test_estimate_head_pose_cues_frontal():
    """Verify frontal face landmark cues estimate near-zero yaw and pitch."""
    _, landmarks, _ = _make_dummy_frame_and_landmarks(yaw_shift=0.0, pitch_shift=0.0)
    cues = estimate_head_pose_cues(landmarks, frame_shape=(200, 200))

    assert cues is not None
    assert isinstance(cues, HeadPoseCues)
    assert abs(cues.yaw_proxy) < 0.05
    assert abs(cues.roll_degrees) < 1.0
    assert cues.eye_distance == pytest.approx(60.0, abs=1.0)


def test_estimate_head_pose_cues_turn_left():
    """Verify head turned left produces positive yaw proxy deflection."""
    _, landmarks, _ = _make_dummy_frame_and_landmarks(yaw_shift=0.5, pitch_shift=0.0)
    cues = estimate_head_pose_cues(landmarks, frame_shape=(200, 200))

    assert cues is not None
    assert cues.yaw_proxy > 0.30


def test_estimate_head_pose_cues_invalid_landmarks():
    """Verify invalid or incomplete landmarks return None safely."""
    assert estimate_head_pose_cues([]) is None
    assert estimate_head_pose_cues([[0, 0], [1, 1]]) is None


def test_extract_eye_patches_and_difference():
    """Verify eye patches are extracted and patch difference computation works."""
    frame, landmarks, _ = _make_dummy_frame_and_landmarks()
    patches = extract_eye_patches(frame, landmarks, target_size=(24, 16))

    assert patches is not None
    left_patch, right_patch = patches
    assert left_patch.shape == (16, 24)
    assert right_patch.shape == (16, 24)

    # Identical patches have difference of 0
    assert compute_patch_difference(left_patch, left_patch) == 0.0

    # Black vs white patch has maximum difference ~ 1.0
    black_patch = np.zeros_like(left_patch)
    white_patch = np.ones_like(left_patch) * 255
    diff = compute_patch_difference(black_patch, white_patch)
    assert diff > 0.95


def test_extract_texture_cues():
    """Verify texture and sharpness metrics on face crop."""
    # Textured noisy image
    np.random.seed(42)
    textured = np.random.randint(0, 256, (60, 60, 3), dtype=np.uint8)
    cues = extract_texture_cues(textured)
    assert cues["laplacian_variance"] > 100.0
    assert "specular_ratio" in cues

    # Completely flat image (zero texture)
    flat = np.ones((60, 60, 3), dtype=np.uint8) * 128
    cues_flat = extract_texture_cues(flat)
    assert cues_flat["laplacian_variance"] == 0.0


def test_temporal_heuristic_detector_accumulation_phase():
    """Verify detector reports UNCERTAIN while accumulating initial required frames."""
    detector = TemporalHeuristicLivenessDetector(required_frames=5)
    frame, landmarks, bbox = _make_dummy_frame_and_landmarks()

    for f in range(1, 5):
        res = detector.is_live(frame, bbox=bbox, landmarks=landmarks, track_id=1)
        assert res.status == LivenessStatus.UNCERTAIN
        assert not res.is_live


def test_temporal_heuristic_detector_static_photo_spoof():
    """Verify detector flags static motionless presentation as SPOOF."""
    detector = TemporalHeuristicLivenessDetector(
        required_frames=5,
        pose_variance_threshold=0.015,
    )
    frame, landmarks, bbox = _make_dummy_frame_and_landmarks(yaw_shift=0.0)

    for _ in range(7):
        res = detector.is_live(frame, bbox=bbox, landmarks=landmarks, track_id=2)

    assert res.status == LivenessStatus.SPOOF
    assert not res.is_live
    assert res.score <= 0.25
    assert "Static presentation attack suspected" in res.reason


def test_temporal_heuristic_detector_dynamic_live_face():
    """Verify detector confirms live face when natural head rotations are observed."""
    detector = TemporalHeuristicLivenessDetector(
        required_frames=5,
        pose_variance_threshold=0.015,
        min_score=0.50,
    )

    shifts = [0.0, 0.10, 0.20, 0.15, -0.10, 0.12, 0.25]
    for shift in shifts:
        frame, landmarks, bbox = _make_dummy_frame_and_landmarks(yaw_shift=shift)
        res = detector.is_live(frame, bbox=bbox, landmarks=landmarks, track_id=3)

    assert res.status == LivenessStatus.LIVE
    assert res.is_live
    assert res.score >= 0.50


def test_temporal_heuristic_detector_reset():
    """Verify reset clears buffer for specific track or all tracks."""
    detector = TemporalHeuristicLivenessDetector(required_frames=5)
    frame, landmarks, bbox = _make_dummy_frame_and_landmarks()

    for _ in range(3):
        detector.is_live(frame, bbox=bbox, landmarks=landmarks, track_id=10)
    assert len(detector._tracks[10]) == 3

    detector.reset(track_id=10)
    assert len(detector._tracks[10]) == 0


def test_challenge_response_detector_completion():
    """Verify ChallengeResponse transitions from PENDING_CHALLENGE to LIVE upon completion."""
    detector = ChallengeResponseLivenessDetector(challenge_timeout_frames=10)
    detector.set_challenge(track_id=1, challenge=ChallengeType.TURN_LEFT)

    # Frame 1: Looking frontal -> Pending
    frame, landmarks, bbox = _make_dummy_frame_and_landmarks(yaw_shift=0.0)
    res1 = detector.is_live(frame, bbox=bbox, landmarks=landmarks, track_id=1)
    assert res1.status == LivenessStatus.PENDING_CHALLENGE
    assert not res1.is_live

    # Frame 2: Turn head left -> Completed!
    frame_left, landmarks_left, bbox_left = _make_dummy_frame_and_landmarks(yaw_shift=0.5)
    res2 = detector.is_live(frame_left, bbox=bbox_left, landmarks=landmarks_left, track_id=1)
    assert res2.status == LivenessStatus.LIVE
    assert res2.is_live
    assert "TURN_LEFT" in res2.reason


def test_challenge_response_detector_timeout():
    """Verify ChallengeResponse transitions to SPOOF upon frame timeout."""
    detector = ChallengeResponseLivenessDetector(challenge_timeout_frames=4)
    detector.set_challenge(track_id=2, challenge=ChallengeType.TURN_LEFT)

    frame, landmarks, bbox = _make_dummy_frame_and_landmarks(yaw_shift=0.0)
    # Process past timeout frames
    res = None
    for _ in range(6):
        res = detector.is_live(frame, bbox=bbox, landmarks=landmarks, track_id=2)

    assert res is not None
    assert res.status == LivenessStatus.SPOOF
    assert not res.is_live
    assert "timed out" in res.reason


def test_passthrough_liveness_detector():
    """Verify Passthrough detector always yields LIVE."""
    detector = PassthroughLivenessDetector()
    frame, landmarks, bbox = _make_dummy_frame_and_landmarks()
    res = detector.is_live(frame, bbox=bbox, landmarks=landmarks)

    assert res.is_live is True
    assert res.score == 1.0
    assert res.status == LivenessStatus.LIVE


def test_liveness_factory_method_selection():
    """Verify factory returns appropriate detector subclasses based on parameter."""
    det_heuristic = get_liveness_detector("heuristic")
    assert isinstance(det_heuristic, TemporalHeuristicLivenessDetector)

    det_challenge = get_liveness_detector("challenge_response")
    assert isinstance(det_challenge, ChallengeResponseLivenessDetector)

    det_passthrough = get_liveness_detector("disabled")
    assert isinstance(det_passthrough, PassthroughLivenessDetector)


def test_liveness_evaluator_audit_record(tmp_path):
    """Verify LivenessEvaluator writes structured JSON records."""
    log_file = tmp_path / "test_audit.log"
    evaluator = LivenessEvaluator(log_path=str(log_file))

    dummy_result = LivenessResult(
        is_live=True,
        score=0.88,
        status=LivenessStatus.LIVE,
        reason="Unit test passed",
        cues={"yaw_std": 0.05},
    )
    evaluator.log_evaluation(
        result=dummy_result,
        track_id=42,
        student_id="STU-TEST",
        session_id="SESS-101",
    )

    assert log_file.is_file()
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "STU-TEST" in content
    assert "SESS-101" in content
    assert '"is_live": true' in content
