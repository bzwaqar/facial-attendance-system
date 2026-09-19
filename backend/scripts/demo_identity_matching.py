"""Demonstration script for FAISS vector search and identity matching."""

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
from app.ml.vector_search import VectorMetadata, VectorSearchEngine
from app.schemas.matching import MatchStatus
from app.schemas.student import StudentCreate
from app.services.enrollment_service import EnrollmentService
from app.services.matching_service import IdentityMatchingService
from app.services.student_service import StudentService

SAMPLE_IMAGE_PATH = backend_dir / "tests" / "data" / "sample_student.jpg"
DEMO_DB_URL = "sqlite+aiosqlite:///demo_matching.db"


async def main() -> None:
    """Run FAISS vector search and identity matching demonstration."""
    setup_logging("INFO")

    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n" + "=" * 75)
    print("  FACIAL RECOGNITION ATTENDANCE SYSTEM - FAISS IDENTITY MATCHING DEMO")
    print("=" * 75)

    if not SAMPLE_IMAGE_PATH.is_file():
        print(f"[ERROR] Sample image missing at {SAMPLE_IMAGE_PATH}")
        sys.exit(1)

    engine = create_async_engine(DEMO_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        # Step 1: Enroll two sample students with multiple face images
        print("\n[1] Registering and enrolling 2 students in database...")
        student_service = StudentService()
        enrollment_service = EnrollmentService()

        # Student A: Grace Hopper
        s_a = await student_service.create_student(
            session,
            StudentCreate(
                student_id="CS-HOOPER-001",
                name="Grace Hopper",
                department="Computer Science",
                semester="Senior",
                section="A",
            ),
        )
        # Student B: Alan Turing
        s_b = await student_service.create_student(
            session,
            StudentCreate(
                student_id="CS-TURING-002",
                name="Alan Turing",
                department="Mathematics & Computing",
                semester="Senior",
                section="B",
            ),
        )
        await session.commit()

        # Enroll 5 reference images for Grace Hopper using sample image variations
        base_img = cv2.imread(str(SAMPLE_IMAGE_PATH))
        images_a = []
        for i in range(1, 6):
            altered = cv2.convertScaleAbs(base_img, alpha=1.0, beta=(i - 3) * 3)
            _, buf = cv2.imencode(".jpg", altered)
            images_a.append((f"hopper_{i}.jpg", buf.tobytes()))

        await enrollment_service.enroll_student_faces(session, "CS-HOOPER-001", images_a)

        # For Alan Turing, add synthetic distinct orthogonal embeddings for demo
        from app.db.models.face_embedding import FaceEmbedding
        np.random.seed(2026)
        turing_basis = np.random.randn(512).astype(np.float32)
        turing_basis /= float(np.linalg.norm(turing_basis))

        for j in range(5):
            t_vec = turing_basis + 0.02 * np.random.randn(512).astype(np.float32)
            t_vec /= float(np.linalg.norm(t_vec))
            emb = FaceEmbedding(
                student_id=s_b.id,
                embedding=b"",
                model_name="ArcFace",
                model_version=settings.INSIGHTFACE_MODEL_NAME,
                quality_score=0.95,
            )
            emb.set_vector(t_vec.tolist())
            session.add(emb)

        await session.commit()
        print("    [+] Successfully enrolled 5 vectors for 'Grace Hopper' (CS-HOOPER-001)")
        print("    [+] Successfully enrolled 5 vectors for 'Alan Turing' (CS-TURING-002)")

        # Step 2: Build / Populate FAISS Index from Database
        print("\n[2] Rebuilding FAISS index from stored database embeddings...")
        matching_service = IdentityMatchingService()
        total_indexed = await matching_service.rebuild_index_from_db(session)
        print(f"    [+] Loaded and indexed {total_indexed} total face vectors into FAISS IndexFlatIP(512).")
        print(f"    [+] Active recognition threshold: {settings.FACE_SIMILARITY_THRESHOLD:.2f}")

        # Step 3: Test Known Identity Query (Live image matching Grace Hopper)
        print("\n[3] Querying with live test image of 'Grace Hopper'...")
        query_result = matching_service.match_live_image(base_img, threshold=0.60, top_k=5)

        print(f"    • Status        : {query_result.status.value}")
        print(f"    • Matched?      : {query_result.matched}")
        print(f"    • Identified ID : {query_result.student_id}")
        print(f"    • Similarity    : {query_result.similarity_score:.4f} (Threshold: {query_result.threshold_used:.2f})")
        print(f"    • Search Latency: {query_result.search_time_ms:.3f} ms")
        print("    • Top-5 Candidate Breakdown:")
        for rank, cand in enumerate(query_result.candidates, start=1):
            print(
                f"        Rank {rank}: Student '{cand.student_id}' | "
                f"Similarity: {cand.similarity_score:.4f} | Vector #{cand.vector_id}"
            )

        # Step 4: Test Unknown Face Query (Unregistered person embedding)
        print("\n[4] Querying with an unknown / unregistered person face embedding...")
        np.random.seed(9999)
        unknown_vec = np.random.randn(512).astype(np.float32)
        unknown_vec /= float(np.linalg.norm(unknown_vec))

        unknown_result = matching_service.match_face_embedding(unknown_vec, threshold=0.60, top_k=3)

        print(f"    • Status        : {unknown_result.status.value}")
        print(f"    • Matched?      : {unknown_result.matched}")
        print(f"    • Identified ID : {unknown_result.student_id} (Never forces a match)")
        print(f"    • Best Score    : {unknown_result.similarity_score:.4f} (Threshold: {unknown_result.threshold_used:.2f})")
        print(f"    • Message       : {unknown_result.message}")

        # Step 5: Test Empty Index Handling
        print("\n[5] Demonstrating search behavior against an empty index...")
        empty_vector_engine = VectorSearchEngine(dimension=512)
        empty_result = empty_vector_engine.search(unknown_vec)
        print(f"    • Status        : {empty_result.status.value}")
        print(f"    • Matched?      : {empty_result.matched}")
        print(f"    • Message       : {empty_result.message}")

    # Clean up demo database file
    await engine.dispose()
    db_file = backend_dir / "demo_matching.db"
    if db_file.exists():
        db_file.unlink()

    print("\n" + "=" * 75)
    print("  FAISS IDENTITY MATCHING DEMO COMPLETED SUCCESSFULLY")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
