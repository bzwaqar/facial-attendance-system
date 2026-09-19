"""Demonstration script for the Student Biometric Facial Enrollment Pipeline."""

import asyncio
import sys
from pathlib import Path

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import cv2
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.base import Base
from app.schemas.student import StudentCreate
from app.services.enrollment_service import EnrollmentService
from app.services.exceptions import DuplicateStudentError, FaceEnrollmentValidationError
from app.services.student_service import StudentService

SAMPLE_IMAGE_PATH = backend_dir / "tests" / "data" / "sample_student.jpg"
DEMO_DB_URL = "sqlite+aiosqlite:///demo_enrollment.db"


async def main() -> None:
    """Run full student enrollment pipeline demo."""
    setup_logging("INFO")

    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n" + "=" * 75)
    print("  FACIAL RECOGNITION ATTENDANCE SYSTEM - STUDENT ENROLLMENT PIPELINE DEMO")
    print("=" * 75)

    if not SAMPLE_IMAGE_PATH.is_file():
        print(f"[ERROR] Sample image missing at {SAMPLE_IMAGE_PATH}")
        sys.exit(1)

    engine = create_async_engine(DEMO_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        # Step 1: Create Student
        print("\n[1] Registering a new student in the database...")
        student_data = StudentCreate(
            student_id="DEMO-2026-001",
            name="Alexander Graham",
            department="Computer Science & AI",
            semester="6th",
            section="A1",
        )
        try:
            student = await StudentService.create_student(session, student_data)
            await session.commit()
            print(f"    [+] Student registered: {student.name} (Student ID: {student.student_id}, PK: {student.id})")
        except DuplicateStudentError:
            print(f"    [i] Student '{student_data.student_id}' already exists, continuing demo...")

        # Step 2: Prepare 5 facial images for biometric enrollment
        print("\n[2] Preparing 5 facial portrait images from reference camera feed...")
        base_img = cv2.imread(str(SAMPLE_IMAGE_PATH))
        images = []
        for i in range(1, 6):
            # Simulate slight exposure variations in natural capture
            altered = cv2.convertScaleAbs(base_img, alpha=1.0, beta=(i - 3) * 4)
            _, buf = cv2.imencode(".jpg", altered)
            images.append((f"capture_angle_{i}.jpg", buf.tobytes()))
            print(f"    • Image #{i}: capture_angle_{i}.jpg ({len(buf.tobytes()) / 1024:.1f} KB)")

        # Step 3: Run Enrollment Service
        print("\n[3] Executing EnrollmentService.enroll_student_faces()...")
        print("    - SCRFD Face Detection -> Single Face Enforcement -> Quality Filtering -> ArcFace 512-d")
        enrollment_service = EnrollmentService()

        result = await enrollment_service.enroll_student_faces(
            session=session,
            student_id="DEMO-2026-001",
            images=images,
            min_images=5,
            max_images=10,
            replace_existing=True,
        )
        await session.commit()

        print(f"\n[4] Enrollment Result Summary:")
        print(f"    • Total Submitted : {result.total_submitted}")
        print(f"    • Total Enrolled  : {result.total_enrolled}")
        print(f"    • Biometric Model : {result.model_name} ({result.model_version})")
        print(f"    • Status Message  : {result.message}")

        print("\n[5] Per-Image Diagnostic Details:")
        for diag in result.images_summary:
            print(
                f"    [+] {diag.filename:<22} | Status: {diag.status.upper():<8} | "
                f"Conf: {diag.confidence:.4f} | Quality Score: {diag.quality_score:.4f} | "
                f"BBox: {[round(c, 1) for c in (diag.bbox or [])]}"
            )

        # Step 4: Verification of Database Persistence
        print("\n[6] Querying student and verifying stored embeddings from database...")
        persisted_student = await StudentService.get_student_by_student_id(
            session,
            "DEMO-2026-001",
            load_embeddings=True,
        )
        assert persisted_student is not None
        print(f"    - Student           : {persisted_student.name} ({persisted_student.student_id})")
        print(f"    - Enrolled Vectors  : {len(persisted_student.face_embeddings)} records")

        for idx, emb in enumerate(persisted_student.face_embeddings, start=1):
            vec = emb.get_vector()
            norm = float(np.linalg.norm(np.array(vec, dtype=np.float32)))
            print(
                f"      - Embedding #{idx}: Model: {emb.model_name}:{emb.model_version} | "
                f"Dim: {len(vec)} | L2 Norm: {norm:.6f} | "
                f"Vector[0:4]: {[round(v, 4) for v in vec[:4]]}..."
            )

        # Step 5: Test Quality Rejection Demonstration
        print("\n[7] Demonstrating rejection on invalid image (blank image without face)...")
        bad_images = images[:4]
        blank = np.zeros((400, 400, 3), dtype=np.uint8)
        _, blank_buf = cv2.imencode(".jpg", blank)
        bad_images.append(("blank_test_image.jpg", blank_buf.tobytes()))

        try:
            await enrollment_service.enroll_student_faces(
                session=session,
                student_id="DEMO-2026-001",
                images=bad_images,
            )
            print("    [!] Error: Expected validation rejection but succeeded.")
        except FaceEnrollmentValidationError as exc:
            print(f"    [+] Correctly rejected: {exc.message}")
            rejected_diags = [d for d in exc.diagnostics if d.status == "rejected"]
            for d in rejected_diags:
                print(f"      - Rejected File: {d.filename} | Reason: {d.reason}")

    # Clean up demo sqlite file
    await engine.dispose()
    demo_db_path = backend_dir / "demo_enrollment.db"
    if demo_db_path.exists():
        demo_db_path.unlink()

    print("\n" + "=" * 75)
    print("  DEMO COMPLETED SUCCESSFULLY")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
