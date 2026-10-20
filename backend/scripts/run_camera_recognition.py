"""Standalone real-time camera recognition runner with live video display and HUD."""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import cv2
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.camera.capture import CameraCapture
from app.camera.pipeline import CameraRecognitionPipeline
from app.core.config import settings
from app.core.logging import setup_logging
from app.db.models.student import Student
from app.db.session import AsyncSessionLocal
from app.schemas.attendance import AttendanceMarkRequest, AttendanceStatus
from app.services.attendance_service import AttendanceService
from app.services.matching_service import get_identity_matching_service

logger = logging.getLogger(__name__)


async def fetch_student_names(session: AsyncSession):
    """Retrieve mapping of student_id to full student name."""
    query = select(Student.student_id, Student.name).where(Student.is_active.is_(True))
    result = await session.execute(query)
    return {row[0]: row[1] for row in result.all()}


async def record_attendance_event(
    session_id: str,
    student_id: str,
    similarity: float,
    obs_count: int,
    is_live: Optional[bool] = None,
    liveness_score: Optional[float] = None,
) -> None:
    """Record verified attendance asynchronously via AttendanceService."""
    try:
        async with AsyncSessionLocal() as db:
            service = AttendanceService(db=db)
            req = AttendanceMarkRequest(
                session_id=session_id,
                student_id=student_id,
                similarity_score=similarity,
                observation_count=obs_count,
                is_confirmed=True,
                is_live=is_live,
                liveness_score=liveness_score,
            )
            res = await service.mark_attendance(req)
            if res.status == AttendanceStatus.PRESENT:
                print(f"\n>>> [ATTENDANCE SUCCESS] {res.student_name} ({res.student_id}) marked PRESENT in session '{session_id}' <<<\n")
            elif res.status == AttendanceStatus.ALREADY_MARKED:
                pass  # Gracefully ignore already marked
            else:
                logger.info("Attendance not marked for %s: %s (%s)", student_id, res.status, res.message)
    except Exception as exc:
        logger.error("Failed to record attendance for %s in %s: %s", student_id, session_id, exc)



async def init_pipeline(recognition_interval: int, threshold: float):
    """Initialize FAISS index and populate student names from PostgreSQL."""
    matching_service = get_identity_matching_service()
    student_names = {}

    try:
        async with AsyncSessionLocal() as session:
            count = await matching_service.rebuild_index_from_db(session)
            student_names = await fetch_student_names(session)
            print(f"[+] Loaded {count} reference face embeddings for {len(student_names)} active students.")
    except Exception as exc:
        print(f"[!] Warning: Could not connect to database ({exc}). Running with current in-memory index.")

    pipeline = CameraRecognitionPipeline(
        matching_service=matching_service,
        recognition_interval=recognition_interval,
        threshold=threshold,
    )
    pipeline.set_student_names(student_names)
    return pipeline


def main() -> None:
    """CLI entry point for live camera face recognition."""
    parser = argparse.ArgumentParser(
        description="Real-Time Facial Recognition Attendance System (Camera Feed)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=settings.CAMERA_INDEX,
        help=f"Camera device index (default: {settings.CAMERA_INDEX}).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional path to video file input instead of live webcam.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=settings.CAMERA_FPS,
        help=f"Target capture frame rate (default: {settings.CAMERA_FPS}).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=settings.RECOGNITION_INTERVAL_FRAMES,
        help=f"Frames between ArcFace recognition passes per face (default: {settings.RECOGNITION_INTERVAL_FRAMES}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.FACE_SIMILARITY_THRESHOLD,
        help=f"Cosine similarity recognition threshold (default: {settings.FACE_SIMILARITY_THRESHOLD}).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode without opening GUI window (for servers / CI).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process before exiting (useful for testing).",
    )
    parser.add_argument(
        "--attendance-session",
        "--session-id",
        dest="attendance_session",
        type=str,
        default=None,
        help="Optional session_id to automatically record attendance for confirmed recognitions.",
    )
    args = parser.parse_args()

    setup_logging(settings.LOG_LEVEL)


    print("\n" + "=" * 75)
    print("  FACIAL RECOGNITION ATTENDANCE SYSTEM - REAL-TIME CAMERA RECOGNITION")
    print("=" * 75)
    print(f"  - Video Source        : {args.video if args.video else f'Camera #{args.camera}'}")
    print(f"  - Target FPS          : {args.fps}")
    print(f"  - Recognition Interval: Every {args.interval} frames")
    print(f"  - Similarity Threshold: {args.threshold:.2f} (Empirical baseline)")
    print(f"  - Headless Mode       : {args.headless}")
    print("=" * 75 + "\n")

    # Step 1: Initialize Pipeline
    print("[1] Initializing Biometric Models and FAISS Index...")
    pipeline = asyncio.run(init_pipeline(args.interval, args.threshold))

    # Step 2: Initialize Video Capture
    source = args.video if args.video else args.camera
    print(f"[2] Opening Video Source: {source}...")
    capture = CameraCapture(source=source, fps=args.fps)

    if not capture.open():
        print(f"[ERROR] Could not open video source: {source}")
        sys.exit(1)

    print("\n[3] Camera Stream Running. Press 'q' or ESC in display window (or Ctrl+C) to quit.\n")

    is_shutting_down = False

    def handle_signal(sig, frame):
        nonlocal is_shutting_down
        print("\n[!] Shutdown signal received. Exiting cleanly...")
        is_shutting_down = True

    signal.signal(signal.SIGINT, handle_signal)

    frame_num = 0
    window_name = "Facial Recognition Attendance System"

    try:
        while not is_shutting_down:
            success, frame = capture.read_frame()
            if not success or frame is None:
                print("[i] Video stream ended or camera disconnected.")
                break

            frame_num += 1

            # Process frame through detection -> tracking -> periodic recognition -> temporal verification
            result = pipeline.process_frame(frame)

            # Display via OpenCV window if not headless
            if not args.headless:
                try:
                    cv2.imshow(window_name, result.annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):  # 'q' or ESC
                        print("\n[i] User pressed quit key. Shutting down...")
                        break
                except Exception as gui_err:
                    print(f"[!] GUI Display error ({gui_err}). Switching to headless mode.")
                    args.headless = True

            # Periodic console telemetry
            if frame_num % 30 == 0 or result.faces_detected_count > 0:
                confirmed_names = [
                    s.confirmed_student_id
                    for s in result.temporal_states
                    if s and s.confirmed_student_id
                ]
                print(
                    f"Frame {frame_num:5d} | FPS: {result.fps:4.1f} | "
                    f"Detected: {result.faces_detected_count} | Active Tracks: {len(result.tracks)} | "
                    f"Recog Executed: {result.recognitions_executed_count} | "
                    f"Confirmed: {confirmed_names}"
                )

            # If attendance session is configured, record attendance for confirmed recognitions
            if args.attendance_session and result.temporal_states:
                for s in result.temporal_states:
                    if s:
                        if s.is_spoof:
                            print(f"\n>>> [ANTI-SPOOF ALERT] Track #{s.track_id} flagged as SPOOF ({s.liveness_reason}) - Attendance blocked! <<<\n")
                        elif getattr(s, "confirmed_student_id", None):
                            asyncio.run(
                                record_attendance_event(
                                    session_id=args.attendance_session,
                                    student_id=s.confirmed_student_id,
                                    similarity=s.confidence_score,
                                    obs_count=s.total_observations,
                                    is_live=s.is_live,
                                    liveness_score=s.liveness_score,
                                )
                            )

            if args.max_frames and frame_num >= args.max_frames:
                print(f"[i] Reached max frames limit ({args.max_frames}). Stopping.")
                break


    finally:
        print("\n[4] Releasing camera resources and closing windows...")
        capture.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[+] Camera recognition pipeline terminated cleanly.\n")


if __name__ == "__main__":
    main()
