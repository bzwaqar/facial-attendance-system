"""Unit tests for lightweight multi-face IoU tracker."""

import pytest

from app.camera.tracker import FaceTracker, TrackedFace, compute_iou
from app.ml.models import DetectedFace


def test_compute_iou() -> None:
    """Verify exact Intersection-over-Union bounding box calculations."""
    boxA = [100.0, 100.0, 200.0, 200.0]  # Area = 10000

    # 1. Identical boxes -> IoU = 1.0
    assert abs(compute_iou(boxA, boxA) - 1.0) < 1e-5

    # 2. Non-overlapping boxes -> IoU = 0.0
    box_disjoint = [300.0, 300.0, 400.0, 400.0]
    assert compute_iou(boxA, box_disjoint) == 0.0

    # 3. 50% horizontal overlap
    box_half = [150.0, 100.0, 250.0, 200.0]
    # Inter: [150, 100, 200, 200] -> area 5000. Union: 10000 + 10000 - 5000 = 15000 -> IoU = 5000/15000 = 1/3
    expected = 5000.0 / 15000.0
    assert abs(compute_iou(boxA, box_half) - expected) < 1e-4


def test_face_tracker_single_face_trajectory() -> None:
    """Verify that a moving face maintains the same track_id across frames."""
    tracker = FaceTracker(iou_threshold=0.30, max_lost_frames=5)

    # Frame 1: Face at (100, 100, 200, 200)
    det1 = [DetectedFace(bbox=[100.0, 100.0, 200.0, 200.0], confidence=0.95, embedding=None)]
    tracks1 = tracker.update(det1)
    assert len(tracks1) == 1
    track_id = tracks1[0].track_id
    assert tracks1[0].frames_tracked == 1

    # Frame 2: Face shifted slightly to (110, 105, 210, 205)
    det2 = [DetectedFace(bbox=[110.0, 105.0, 210.0, 205.0], confidence=0.94, embedding=None)]
    tracks2 = tracker.update(det2)
    assert len(tracks2) == 1
    assert tracks2[0].track_id == track_id  # Track ID must persist
    assert tracks2[0].frames_tracked == 2
    assert tracks2[0].bbox == [110.0, 105.0, 210.0, 205.0]

    # Frame 3: Face shifted to (120, 110, 220, 210)
    det3 = [DetectedFace(bbox=[120.0, 110.0, 220.0, 210.0], confidence=0.96, embedding=None)]
    tracks3 = tracker.update(det3)
    assert len(tracks3) == 1
    assert tracks3[0].track_id == track_id
    assert tracks3[0].frames_tracked == 3


def test_face_tracker_multi_face_tracking() -> None:
    """Verify that multiple faces in a frame receive distinct track IDs."""
    tracker = FaceTracker(iou_threshold=0.30)

    # Two distinct faces in the frame
    faceA = DetectedFace(bbox=[50.0, 50.0, 150.0, 150.0], confidence=0.90, embedding=None)
    faceB = DetectedFace(bbox=[300.0, 200.0, 400.0, 300.0], confidence=0.92, embedding=None)

    tracks = tracker.update([faceA, faceB])
    assert len(tracks) == 2
    assert tracks[0].track_id != tracks[1].track_id

    # Second frame with both faces shifted slightly
    faceA_next = DetectedFace(bbox=[55.0, 52.0, 155.0, 152.0], confidence=0.89, embedding=None)
    faceB_next = DetectedFace(bbox=[305.0, 198.0, 405.0, 298.0], confidence=0.93, embedding=None)

    tracks_next = tracker.update([faceA_next, faceB_next])
    assert len(tracks_next) == 2
    id_map = {t.track_id: t for t in tracks_next}
    assert tracks[0].track_id in id_map
    assert tracks[1].track_id in id_map


def test_face_tracker_lost_and_pruned() -> None:
    """Verify that a missing face is aged and pruned after max_lost_frames."""
    tracker = FaceTracker(max_lost_frames=3)

    # Frame 1: Face appears
    det = [DetectedFace(bbox=[100.0, 100.0, 200.0, 200.0], confidence=0.95, embedding=None)]
    tracker.update(det)
    assert tracker.active_tracks_count == 1

    # Frames 2, 3: Face absent
    tracks_f2 = tracker.update([])
    assert len(tracks_f2) == 0
    assert tracker.active_tracks_count == 1  # Still kept in active_tracks as lost_frames=1

    tracks_f3 = tracker.update([])
    assert tracker.active_tracks_count == 1  # lost_frames=2

    # Frame 4: lost_frames=3 -> Still alive
    tracker.update([])
    assert tracker.active_tracks_count == 1

    # Frame 5: lost_frames=4 > max_lost_frames (3) -> Pruned
    tracker.update([])
    assert tracker.active_tracks_count == 0


def test_should_run_recognition_interval() -> None:
    """Verify that recognition interval is triggered on frame 1 and every N frames."""
    tracker = FaceTracker()
    track = TrackedFace(track_id=1, bbox=[100.0, 100.0, 200.0, 200.0], confidence=0.9)

    # Track never recognized -> must run recognition
    assert tracker.should_run_recognition(track, interval=10) is True

    # Mark recognized at frame 10
    tracker._frame_count = 10
    tracker.mark_recognized(track.track_id)
    track.last_recognition_frame = 10

    # Frames 11 to 19 -> should NOT run recognition
    tracker._frame_count = 15
    assert tracker.should_run_recognition(track, interval=10) is False

    tracker._frame_count = 19
    assert tracker.should_run_recognition(track, interval=10) is False

    # Frame 20 (interval elapsed: 20 - 10 = 10) -> MUST run recognition
    tracker._frame_count = 20
    assert tracker.should_run_recognition(track, interval=10) is True
