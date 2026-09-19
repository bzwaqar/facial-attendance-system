"""Demonstration script for the InsightFace face-processing pipeline."""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.logging import setup_logging
from app.ml import FaceEngine, get_face_engine
from app.ml.loader import get_model_loader


def main() -> None:
    """Run face detection, alignment, and ArcFace embedding extraction demo."""
    parser = argparse.ArgumentParser(description="Demo Face-Processing Pipeline (SCRFD + ArcFace)")
    parser.add_argument(
        "--image",
        type=str,
        default=str(backend_dir / "tests" / "data" / "sample_student.jpg"),
        help="Path to image file to process.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Minimum face detection confidence threshold (default: 0.50).",
    )
    args = parser.parse_args()

    setup_logging("INFO")
    print("\n" + "=" * 70)
    print("  FACIAL RECOGNITION ATTENDANCE SYSTEM - BIOMETRIC PIPELINE DEMO")
    print("=" * 70)

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"\n[ERROR] Test image not found at: {image_path}")
        sys.exit(1)

    print(f"\n[1] Target image: {image_path.resolve()}")

    # 1. Initialize Engine (Model loaded ONCE into memory)
    print("\n[2] Initializing FaceEngine (loading pretrained InsightFace models)...")
    loader = get_model_loader()
    engine = FaceEngine(loader=loader)

    print(f"    - Execution Providers: {loader.active_providers}")
    print(f"    - Model loaded: {loader.is_loaded}")

    # 2. First inference pass
    print("\n[3] Executing face processing pipeline on input image...")
    start = time.perf_counter()
    result = engine.process_image(image_path, confidence_threshold=args.threshold)
    elapsed = (time.perf_counter() - start) * 1000.0

    print(f"    - Faces Detected: {result.total_detected}")
    print(f"    - Total Inference Time: {result.processing_time_ms:.1f} ms")

    if not result.has_faces:
        print("\n[!] No faces detected exceeding threshold.")
        return

    # 3. Print structured details for each detected face
    for idx, face in enumerate(result.faces, start=1):
        norm = float(np.linalg.norm(face.embedding))
        print(f"\n--- Detected Face #{idx} ---")
        print(f"  • Bounding Box (x1, y1, x2, y2) : {face.bbox}")
        print(f"  • Detection Confidence (SCRFD) : {face.confidence:.4f}")
        print(f"  • Facial Landmarks Count        : {len(face.landmarks) if face.landmarks else 0} points")
        print(f"  • ArcFace Embedding Dimension  : {face.embedding_dim} floats")
        print(f"  • Embedding L2 Unit Norm        : {norm:.6f} (target: 1.000000)")
        print(f"  • Embedding Vector (first 5)    : {[round(float(v), 5) for v in face.embedding[:5]]}...")

    # 4. Second inference pass to demonstrate model reuse without reloading
    print("\n[4] Running second inference pass to demonstrate zero-reload model reuse...")
    start2 = time.perf_counter()
    result2 = engine.process_image(image_path, confidence_threshold=args.threshold)
    elapsed2 = (time.perf_counter() - start2) * 1000.0
    print(f"    - Second pass completed in: {elapsed2:.1f} ms (cached model reused)")

    print("\n" + "=" * 70)
    print("  DEMO COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
