"""Temporal observation collection and identity verification across video frames."""

import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional
import numpy as np

from app.core.config import settings
from app.schemas.matching import IdentityMatchResult, MatchStatus

logger = logging.getLogger(__name__)


class TemporalIdentityState(str, Enum):
    """Temporal lifecycle verification states for a tracked face."""

    VERIFYING = "VERIFYING"  # Gathering initial candidate observations
    CONFIRMED = "CONFIRMED"  # Consistency verified across required observation count
    UNKNOWN = "UNKNOWN"      # Observations consistently fall below recognition threshold


@dataclass
class IdentityObservation:
    """A single recognition observation sample recorded for a tracked face."""

    timestamp: float
    student_id: Optional[str]
    student_pk: Optional[int]
    student_name: Optional[str]
    similarity: float
    matched: bool


@dataclass
class TrackTemporalState:
    """Consolidated temporal verification status for an ongoing face track."""

    track_id: int
    state: TemporalIdentityState = TemporalIdentityState.VERIFYING
    confirmed_student_id: Optional[str] = None
    confirmed_student_name: Optional[str] = None
    confirmed_student_pk: Optional[int] = None
    confidence_score: float = 0.0
    is_live: Optional[bool] = None
    liveness_score: Optional[float] = None
    liveness_status: Optional[str] = None
    liveness_reason: Optional[str] = None
    observations: Deque[IdentityObservation] = field(default_factory=deque)

    @property
    def total_observations(self) -> int:
        return len(self.observations)

    @property
    def is_spoof(self) -> bool:
        """True if liveness detector flagged this track as an attack / spoof."""
        return self.liveness_status == "SPOOF" or self.is_live is False

    @property
    def display_label(self) -> str:
        """Formatted label string for visual HUD overlay."""
        if self.is_spoof:
            return f"SPOOF DETECTED ({self.liveness_reason or 'Attack'})"
        elif self.liveness_status == "PENDING_CHALLENGE":
            return f"{self.liveness_reason or 'Challenge Pending'}"
        elif self.state == TemporalIdentityState.CONFIRMED:
            name_str = f" {self.confirmed_student_name}" if self.confirmed_student_name else ""
            live_tag = " [LIVE]" if self.is_live else ""
            return f"ID: {self.confirmed_student_id}{name_str}{live_tag} ({self.confidence_score:.2f})"
        elif self.state == TemporalIdentityState.UNKNOWN:
            return f"UNKNOWN ({self.confidence_score:.2f})"
        else:
            return f"Verifying ({self.total_observations} obs)..."


class TemporalFaceVerifier:
    """Evaluates temporal observation sequences per track before confirming student identity.
    
    Prevents instantaneous false positives or flapping by requiring consistent identification
    across multiple frames before transitioning to CONFIRMED.
    """

    def __init__(
        self,
        window_size: Optional[int] = None,
        min_confirmations: Optional[int] = None,
    ):
        """Initialize temporal face verifier.
        
        Args:
            window_size: Maximum sliding history of observations retained per track.
            min_confirmations: Minimum number of matching observations required to confirm identity.
        """
        self.window_size = window_size or settings.TEMPORAL_WINDOW_SIZE
        self.min_confirmations = min_confirmations or settings.TEMPORAL_MIN_CONFIRMATIONS
        self._tracks: Dict[int, TrackTemporalState] = {}

    def get_track_state(self, track_id: int) -> Optional[TrackTemporalState]:
        """Retrieve current temporal state for a track."""
        return self._tracks.get(track_id)

    def update_liveness(
        self,
        track_id: int,
        is_live: bool,
        score: float,
        status: str,
        reason: str,
    ) -> TrackTemporalState:
        """Update anti-spoofing evaluation for a track and revoke confirmation if spoofed."""
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackTemporalState(
                track_id=track_id,
                observations=deque(maxlen=self.window_size),
            )
        track_state = self._tracks[track_id]
        track_state.is_live = is_live
        track_state.liveness_score = score
        track_state.liveness_status = status
        track_state.liveness_reason = reason

        # Revoke confirmed identity immediately if presentation attack detected
        if not is_live and status == "SPOOF":
            track_state.state = TemporalIdentityState.UNKNOWN
            track_state.confirmed_student_id = None
            track_state.confirmed_student_name = None
            track_state.confirmed_student_pk = None

        return track_state

    def add_observation(
        self,
        track_id: int,
        match_result: IdentityMatchResult,
        student_name: Optional[str] = None,
    ) -> TrackTemporalState:
        """Record a recognition observation and evaluate temporal confirmation criteria.
        
        Args:
            track_id: Persistent track identifier.
            match_result: IdentityMatchResult returned by FAISS search.
            student_name: Optional full student name for visual display.
            
        Returns:
            Updated TrackTemporalState for this track.
        """
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackTemporalState(
                track_id=track_id,
                observations=deque(maxlen=self.window_size),
            )

        track_state = self._tracks[track_id]

        # Record sample
        obs = IdentityObservation(
            timestamp=time.perf_counter(),
            student_id=match_result.student_id if match_result.matched else None,
            student_pk=match_result.student_pk if match_result.matched else None,
            student_name=student_name,
            similarity=match_result.similarity_score,
            matched=match_result.matched,
        )
        track_state.observations.append(obs)

        # Evaluate confirmation criteria across the observation window
        matched_obs = [o for o in track_state.observations if o.matched and o.student_id is not None]
        id_counts = Counter(o.student_id for o in matched_obs)

        # Check for consistent confirmation:
        # At least min_confirmations matching the exact same student_id
        confirmed_id: Optional[str] = None
        for candidate_id, count in id_counts.items():
            if count >= self.min_confirmations:
                confirmed_id = candidate_id
                break

        if confirmed_id:
            # Anti-spoofing gate: If liveness verification failed, reject confirmation
            if track_state.is_live is False:
                logger.warning(
                    "Track #%d recognition match for student '%s' suppressed: Liveness failed (%s).",
                    track_id,
                    confirmed_id,
                    track_state.liveness_reason,
                )
                track_state.state = TemporalIdentityState.UNKNOWN
                track_state.confirmed_student_id = None
                track_state.confirmed_student_name = None
                track_state.confirmed_student_pk = None
                return track_state

            matching_scores = [o.similarity for o in matched_obs if o.student_id == confirmed_id]
            avg_score = float(np.mean(matching_scores)) if matching_scores else 0.0

            matching_names = [o.student_name for o in matched_obs if o.student_id == confirmed_id and o.student_name]
            resolved_name = student_name or (matching_names[0] if matching_names else track_state.confirmed_student_name)

            matching_pks = [o.student_pk for o in matched_obs if o.student_id == confirmed_id and o.student_pk is not None]
            resolved_pk = match_result.student_pk or (matching_pks[0] if matching_pks else track_state.confirmed_student_pk)

            track_state.state = TemporalIdentityState.CONFIRMED
            track_state.confirmed_student_id = confirmed_id
            track_state.confirmed_student_name = resolved_name
            track_state.confirmed_student_pk = resolved_pk
            track_state.confidence_score = round(avg_score, 4)
            logger.debug(
                "Track #%d CONFIRMED as student '%s' (avg similarity: %.4f)",
                track_id,
                confirmed_id,
                avg_score,
            )

        elif len(track_state.observations) >= self.min_confirmations:
            # If we have collected enough samples but no single student reaches min_confirmations
            highest_score = max((o.similarity for o in track_state.observations), default=0.0)
            track_state.state = TemporalIdentityState.UNKNOWN
            track_state.confirmed_student_id = None
            track_state.confirmed_student_name = None
            track_state.confidence_score = round(highest_score, 4)
        else:
            # Still gathering observations
            track_state.state = TemporalIdentityState.VERIFYING
            track_state.confidence_score = round(match_result.similarity_score, 4)

        return track_state

    def cleanup_stale_tracks(self, active_track_ids: List[int]) -> None:
        """Remove state for tracks that have terminated or left the camera frame."""
        active_set = set(active_track_ids)
        stale_ids = [tid for tid in self._tracks if tid not in active_set]
        for tid in stale_ids:
            del self._tracks[tid]

    def reset(self) -> None:
        """Clear all temporal observation history."""
        self._tracks.clear()
