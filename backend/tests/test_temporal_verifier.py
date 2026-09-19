"""Unit tests for TemporalFaceVerifier and temporal observation sequences."""

import pytest

from app.camera.temporal import TemporalFaceVerifier, TemporalIdentityState
from app.schemas.matching import IdentityMatchResult, MatchCandidate, MatchStatus


def _create_mock_match_result(
    matched: bool,
    student_id: str | None = None,
    similarity: float = 0.85,
) -> IdentityMatchResult:
    """Helper creating mock IdentityMatchResult."""
    return IdentityMatchResult(
        status=MatchStatus.MATCHED if matched else MatchStatus.UNKNOWN,
        matched=matched,
        student_id=student_id if matched else None,
        similarity_score=similarity,
        threshold_used=0.60,
        candidates=[
            MatchCandidate(
                student_id=student_id or "UNKNOWN",
                similarity_score=similarity,
                vector_id=0,
            )
        ] if student_id else [],
    )


def test_temporal_state_transitions_to_confirmed() -> None:
    """Verify state transitions: VERIFYING -> CONFIRMED upon receiving min_confirmations matching samples."""
    verifier = TemporalFaceVerifier(window_size=5, min_confirmations=3)

    # Observation 1: Match STU-001 -> VERIFYING (only 1 observation)
    res1 = _create_mock_match_result(matched=True, student_id="STU-001", similarity=0.90)
    state1 = verifier.add_observation(track_id=1, match_result=res1, student_name="Alice Smith")
    assert state1.state == TemporalIdentityState.VERIFYING
    assert state1.confirmed_student_id is None

    # Observation 2: Match STU-001 -> VERIFYING (2 observations)
    res2 = _create_mock_match_result(matched=True, student_id="STU-001", similarity=0.92)
    state2 = verifier.add_observation(track_id=1, match_result=res2)
    assert state2.state == TemporalIdentityState.VERIFYING

    # Observation 3: Match STU-001 -> Reaches min_confirmations (3) -> CONFIRMED!
    res3 = _create_mock_match_result(matched=True, student_id="STU-001", similarity=0.94)
    state3 = verifier.add_observation(track_id=1, match_result=res3)
    assert state3.state == TemporalIdentityState.CONFIRMED
    assert state3.confirmed_student_id == "STU-001"
    assert state3.confirmed_student_name == "Alice Smith"
    assert abs(state3.confidence_score - 0.92) < 1e-4  # (0.90 + 0.92 + 0.94) / 3 = 0.92


def test_temporal_state_transitions_to_unknown() -> None:
    """Verify state transitions to UNKNOWN when observations consistently fail threshold."""
    verifier = TemporalFaceVerifier(window_size=5, min_confirmations=3)

    # 3 consecutive unknown / below-threshold observations
    for _ in range(3):
        res = _create_mock_match_result(matched=False, student_id=None, similarity=0.35)
        state = verifier.add_observation(track_id=2, match_result=res)

    assert state.state == TemporalIdentityState.UNKNOWN
    assert state.confirmed_student_id is None
    assert state.confidence_score == 0.35


def test_temporal_inconsistent_flapping_remains_unconfirmed() -> None:
    """Verify that alternating/flapping candidate IDs do not falsely trigger confirmation."""
    verifier = TemporalFaceVerifier(window_size=5, min_confirmations=3)

    # Alternate between STU-001 and STU-002: neither achieves 3 matching samples
    verifier.add_observation(1, _create_mock_match_result(True, "STU-001", 0.70))
    verifier.add_observation(1, _create_mock_match_result(True, "STU-002", 0.65))
    verifier.add_observation(1, _create_mock_match_result(True, "STU-001", 0.72))
    state = verifier.add_observation(1, _create_mock_match_result(True, "STU-002", 0.68))

    # STU-001 has 2 votes, STU-002 has 2 votes (neither has >= 3)
    assert state.state != TemporalIdentityState.CONFIRMED
    assert state.confirmed_student_id is None


def test_cleanup_stale_tracks() -> None:
    """Verify that stale tracks are cleanly removed from verifier history."""
    verifier = TemporalFaceVerifier()

    verifier.add_observation(1, _create_mock_match_result(True, "STU-001"))
    verifier.add_observation(2, _create_mock_match_result(True, "STU-002"))
    assert verifier.get_track_state(1) is not None
    assert verifier.get_track_state(2) is not None

    # Track 1 left the frame; only track 2 is active
    verifier.cleanup_stale_tracks(active_track_ids=[2])
    assert verifier.get_track_state(1) is None
    assert verifier.get_track_state(2) is not None
