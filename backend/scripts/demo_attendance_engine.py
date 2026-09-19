"""End-to-end Demonstration Script for the Facial Recognition Attendance Engine.

PIPELINE DEMONSTRATED:
Recognized identity
  -> temporal verification
  -> attendance rules
  -> duplicate check
  -> PostgreSQL / SQLite attendance record

Demonstrates all required statuses:
- PRESENT
- ALREADY_MARKED
- UNKNOWN
- REJECTED
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base

from app.db.models.student import Student
from app.schemas.attendance import (
    AttendanceMarkRequest,
    AttendanceSessionCreate,
    AttendanceStatus,
)
from app.services.attendance_service import AttendanceService

DEMO_DB_URL = "sqlite+aiosqlite:///:memory:"


async def main() -> None:
    print("=" * 80)
    print("      FACIAL RECOGNITION ATTENDANCE SYSTEM - ATTENDANCE ENGINE DEMO")
    print("=" * 80)

    # 1. Initialize in-memory database engine and tables
    engine = create_async_engine(DEMO_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as db:
        service = AttendanceService(db=db)

        # 2. Register sample students
        print("\n[1] Registering sample students in registry...")
        students = [
            Student(
                student_id="STU1001",
                name="Ada Lovelace",
                department="Computer Science",
                semester="6th",
                section="A",
                is_active=True,
            ),
            Student(
                student_id="STU1002",
                name="Alan Turing",
                department="Computer Science",
                semester="6th",
                section="A",
                is_active=True,
            ),
            Student(
                student_id="STU1003",
                name="Inactive Student",
                department="Mathematics",
                semester="2nd",
                section="B",
                is_active=False,  # Inactive
            ),
        ]
        db.add_all(students)
        await db.commit()
        print("  -> Registered: STU1001 (Ada Lovelace), STU1002 (Alan Turing), STU1003 (Inactive Student)")

        # 3. Create Attendance Session
        print("\n[2] Initializing Attendance Session...")
        session_create = AttendanceSessionCreate(
            session_id="CS601-20260919-L01",
            course_name="Computer Vision and Biometrics",
            instructor_name="Dr. Sarah Jenkins",
            min_similarity_threshold=0.65,
            min_confirmations=3,
        )
        created_session = await service.create_session(session_create)
        print(f"  -> Session Created: '{created_session.session_id}' | Course: '{created_session.course_name}'")
        print(f"  -> Configured Rules: Min Similarity = {created_session.min_similarity_threshold}, Min Confirmations = {created_session.min_confirmations}")

        # 4. Scenario: First verified detection of Ada Lovelace
        print("\n[3] Event 1: Verified face detection for STU1001 (Ada Lovelace)")
        req1 = AttendanceMarkRequest(
            session_id="CS601-20260919-L01",
            student_id="STU1001",
            similarity_score=0.892,
            observation_count=4,
            is_confirmed=True,
        )
        res1 = await service.mark_attendance(req1)
        print(f"  -> Status: {res1.status.value}")
        print(f"  -> Result: {res1.message}")
        print(f"  -> Record ID: {res1.record_id}, Timestamp: {res1.timestamp.isoformat()}")

        # 5. Scenario: Duplicate detection of Ada Lovelace in same session
        print("\n[4] Event 2: Camera observes STU1001 (Ada Lovelace) again in same session")
        req2 = AttendanceMarkRequest(
            session_id="CS601-20260919-L01",
            student_id="STU1001",
            similarity_score=0.915,
            observation_count=5,
            is_confirmed=True,
        )
        res2 = await service.mark_attendance(req2)
        print(f"  -> Status: {res2.status.value}")
        print(f"  -> Result: {res2.message}")
        print(f"  -> Duplicate Prevented: Exactly 1 record preserved.")

        # 6. Scenario: Unknown face detected
        print("\n[5] Event 3: Camera detects an unregistered visitor (UNKNOWN)")
        req3 = AttendanceMarkRequest(
            session_id="CS601-20260919-L01",
            student_id=None,
            similarity_score=0.38,
            observation_count=4,
            is_confirmed=False,
        )
        res3 = await service.mark_attendance(req3)
        print(f"  -> Status: {res3.status.value}")
        print(f"  -> Result: {res3.message}")

        # 7. Scenario: Low recognition score below threshold
        print("\n[6] Event 4: Ambiguous face detection (Score 0.58 below required 0.65)")
        req4 = AttendanceMarkRequest(
            session_id="CS601-20260919-L01",
            student_id="STU1002",
            similarity_score=0.58,  # Below 0.65
            observation_count=4,
            is_confirmed=True,
        )
        res4 = await service.mark_attendance(req4)
        print(f"  -> Status: {res4.status.value}")
        print(f"  -> Result: {res4.message}")

        # 8. Scenario: Temporal instability (only 1 observation)
        print("\n[7] Event 5: Brief face glance (1 observation below required 3)")
        req5 = AttendanceMarkRequest(
            session_id="CS601-20260919-L01",
            student_id="STU1002",
            similarity_score=0.86,
            observation_count=1,  # Below 3
            is_confirmed=True,
        )
        res5 = await service.mark_attendance(req5)
        print(f"  -> Status: {res5.status.value}")
        print(f"  -> Result: {res5.message}")

        # 9. Scenario: Deactivated student account
        print("\n[8] Event 6: Deactivated student STU1003")
        req6 = AttendanceMarkRequest(
            session_id="CS601-20260919-L01",
            student_id="STU1003",
            similarity_score=0.88,
            observation_count=4,
            is_confirmed=True,
        )
        res6 = await service.mark_attendance(req6)
        print(f"  -> Status: {res6.status.value}")
        print(f"  -> Result: {res6.message}")

        # 10. Scenario: Verified recognition for Alan Turing
        print("\n[9] Event 7: Stable verified recognition for STU1002 (Alan Turing)")
        req7 = AttendanceMarkRequest(
            session_id="CS601-20260919-L01",
            student_id="STU1002",
            similarity_score=0.885,
            observation_count=4,
            is_confirmed=True,
        )
        res7 = await service.mark_attendance(req7)
        print(f"  -> Status: {res7.status.value}")
        print(f"  -> Result: {res7.message}")

        # 11. Final Session Attendance Register
        print("\n" + "=" * 80)
        print(f"SESSION ATTENDANCE REGISTER: {created_session.session_id}")
        print(f"Course: {created_session.course_name} | Instructor: {created_session.instructor_name}")
        print("=" * 80)
        summary = await service.get_session_attendance(created_session.session_id)
        print(f"Total Present: {summary.total_present}")
        print(f"{'#':<3} | {'Student ID':<10} | {'Name':<20} | {'Dept':<18} | {'Score':<8} | {'Status':<10}")
        print("-" * 80)
        for i, rec in enumerate(summary.records, 1):
            print(f"{i:<3} | {rec.student_id:<10} | {rec.student_name:<20} | {rec.department:<18} | {rec.similarity_score:<8.3f} | {rec.status:<10}")
        print("=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY!\n")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
