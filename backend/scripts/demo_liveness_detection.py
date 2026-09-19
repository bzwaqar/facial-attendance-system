"""Demonstration script for anti-spoofing and liveness detection layer.

Simulates:
1. Passive presentation attack scenario (e.g. static printed photo held still before camera).
2. Genuine live face presentation scenario with natural micro-movements and head dynamics.
3. Interactive Challenge-Response verification scenario (e.g., Turn Head Left).
4. Structured evaluation audit logging for empirical benchmarking.
"""

import sys
from pathlib import Path
import time
import numpy as np

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.liveness.base import ChallengeType, LivenessStatus
from app.liveness.cues import estimate_head_pose_cues
from app.liveness.detector import (
    ChallengeResponseLivenessDetector,
    TemporalHeuristicLivenessDetector,
)
from app.liveness.evaluator import get_liveness_evaluator
from app.liveness.factory import get_liveness_detector


def generate_synthetic_face_frame(
    yaw_shift: float = 0.0,
    pitch_shift: float = 0.0,
    w: int = 640,
    h: int = 480,
) -> tuple[np.ndarray, list[list[float]], list[float]]:
    """Generate a synthetic frame with 5 facial landmarks for reproducible benchmarking."""
    frame = np.ones((h, w, 3), dtype=np.uint8) * 128

    # Baseline landmark positions (frontal face)
    cx, cy = w // 2, h // 2
    # Eye distance = 100px
    le = [float(cx - 50), float(cy - 40)]
    re = [float(cx + 50), float(cy - 40)]
    # Nose tip position relative to eyes models 3D yaw and pitch deflection
    nose = [float(cx + yaw_shift * 50), float(cy + pitch_shift * 40)]
    lm = [float(cx - 35), float(cy + 40)]
    rm = [float(cx + 35), float(cy + 40)]

    landmarks = [le, re, nose, lm, rm]
    bbox = [float(cx - 80), float(cy - 90), float(cx + 80), float(cy + 90)]
    return frame, landmarks, bbox


def demo_passive_static_photo_rejection():
    """Scenario 1: Detect and reject static presentation attack (e.g. photo on phone/paper)."""
    print("=" * 75)
    print("SCENARIO 1: Passive Anti-Spoofing vs Static Photo Attack")
    print("=" * 75)
    print("[*] Simulating 8 frames of a completely stationary printed photograph...")

    detector = TemporalHeuristicLivenessDetector(
        required_frames=5,
        pose_variance_threshold=0.03,
        min_score=0.50,
    )

    for frame_idx in range(1, 9):
        # Stationary photo: zero yaw and pitch deflection
        frame, landmarks, bbox = generate_synthetic_face_frame(yaw_shift=0.0, pitch_shift=0.0)
        res = detector.is_live(frame, bbox=bbox, landmarks=landmarks, track_id=101)

        print(
            f"  Frame {frame_idx:2d} | Status: {res.status.value:17s} | "
            f"Live: {str(res.is_live):5s} | Score: {res.score:.3f} | {res.reason}"
        )

    assert res.status == LivenessStatus.SPOOF
    assert not res.is_live
    print("\n[+] SUCCESS: Static photograph was correctly flagged as SPOOF Presentation Attack!\n")


def demo_passive_live_face_acceptance():
    """Scenario 2: Verify genuine live face with natural micro-movements."""
    print("=" * 75)
    print("SCENARIO 2: Passive Anti-Spoofing with Genuine Live Face Movements")
    print("=" * 75)
    print("[*] Simulating 8 frames of a live human with natural micro-rotations and breathing...")

    detector = TemporalHeuristicLivenessDetector(
        required_frames=5,
        pose_variance_threshold=0.015,
        min_score=0.50,
    )

    # Sequence with realistic natural head rotation and micro-movements
    natural_shifts = [0.0, 0.08, 0.18, 0.12, -0.08, 0.15, 0.22, 0.10]

    for frame_idx, shift in enumerate(natural_shifts, start=1):
        frame, landmarks, bbox = generate_synthetic_face_frame(yaw_shift=shift, pitch_shift=shift * 0.5)
        res = detector.is_live(frame, bbox=bbox, landmarks=landmarks, track_id=202)

        print(
            f"  Frame {frame_idx:2d} | Status: {res.status.value:17s} | "
            f"Live: {str(res.is_live):5s} | Score: {res.score:.3f} | {res.reason}"
        )

    assert res.status == LivenessStatus.LIVE
    assert res.is_live
    print("\n[+] SUCCESS: Genuine presentation with dynamic cues was correctly verified as LIVE!\n")


def demo_interactive_challenge_response():
    """Scenario 3: Active challenge-response verification (e.g. Turn Head Left)."""
    print("=" * 75)
    print("SCENARIO 3: Active Interactive Challenge-Response")
    print("=" * 75)

    challenge_detector = ChallengeResponseLivenessDetector(challenge_timeout_frames=10)
    # Assign TURN_LEFT challenge
    challenge_detector.set_challenge(track_id=303, challenge=ChallengeType.TURN_LEFT)

    print("[*] Issued challenge: TURN_LEFT. Monitoring subject response over frames...")

    # Frames 1-3: User is still looking forward
    for f in range(1, 4):
        frame, landmarks, bbox = generate_synthetic_face_frame(yaw_shift=0.0)
        res = challenge_detector.is_live(frame, bbox=bbox, landmarks=landmarks, track_id=303)
        print(f"  Frame {f:2d} | Status: {res.status.value:17s} | {res.reason}")
        assert res.status == LivenessStatus.PENDING_CHALLENGE

    # Frame 4: User turns head left (yaw_shift = 0.25)
    frame, landmarks, bbox = generate_synthetic_face_frame(yaw_shift=0.25)
    res = challenge_detector.is_live(frame, bbox=bbox, landmarks=landmarks, track_id=303)
    print(f"  Frame  4 | Status: {res.status.value:17s} | {res.reason}")

    assert res.status == LivenessStatus.LIVE
    assert res.is_live
    print("\n[+] SUCCESS: Active physical challenge verified and accepted!\n")


def demo_audit_evaluator_logging():
    """Scenario 4: Verify audit log generation."""
    print("=" * 75)
    print("SCENARIO 4: Evaluation and Security Audit Logging")
    print("=" * 75)

    evaluator = get_liveness_evaluator()
    print(f"[+] Evaluation audit log destination: {evaluator.log_path.resolve()}")

    if evaluator.log_path.is_file():
        with open(evaluator.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"[+] Total audit records logged in file: {len(lines)}")
        if lines:
            print(f"[+] Latest audit record preview:\n    {lines[-1].strip()}")
    else:
        print("[!] Note: Audit log will be created upon first evaluation event.")


def main():
    print("\n" + "=" * 75)
    print(" FACIAL RECOGNITION ATTENDANCE SYSTEM - ANTI-SPOOFING & LIVENESS DEMO")
    print("=" * 75 + "\n")

    demo_passive_static_photo_rejection()
    demo_passive_live_face_acceptance()
    demo_interactive_challenge_response()
    demo_audit_evaluator_logging()

    print("\n" + "=" * 75)
    print(" ANTI-SPOOFING & LIVENESS LAYER DEMONSTRATION COMPLETE")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
