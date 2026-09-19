"""Standalone biometric evaluation and hardware profiling runner.

Measures:
1. Genuine vs Impostor Verification Accuracy
2. False Acceptance Rate (FAR) & False Rejection Rate (FRR)
3. Unknown-Person Rejection (True Negative Rate)
4. Similarity Distribution Analysis (Mean, Std, Quartiles)
5. Threshold Sensitivity Sweeps & EER
6. Real Hardware Latency (SCRFD detection, ArcFace embedding, FAISS search, Liveness check) & FPS
"""

import json
import os
from pathlib import Path
import sys

import cv2

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.evaluation.benchmark import BiometricBenchmarkRunner
from app.ml.engine import get_face_engine

SAMPLE_IMAGE_PATH = backend_dir / "tests" / "data" / "sample_student.jpg"
OUTPUT_REPORT_PATH = backend_dir / "logs" / "evaluation_report.json"


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format tabular text for terminal presentation."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    data_lines = [
        " | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(row))
        for row in rows
    ]
    return f"{header_line}\n{sep_line}\n" + "\n".join(data_lines)


def main():
    print("\n" + "=" * 80)
    print("      FACIAL RECOGNITION ATTENDANCE SYSTEM - BIOMETRIC EVALUATION ENGINE")
    print("=" * 80 + "\n")

    print("[1] Initializing Pretrained InsightFace Biometric Engine (buffalo_sc)...")
    face_engine = get_face_engine()

    test_image = None
    if SAMPLE_IMAGE_PATH.is_file():
        test_image = cv2.imread(str(SAMPLE_IMAGE_PATH))
        print(f"[+] Loaded validation portrait: {SAMPLE_IMAGE_PATH.name} ({test_image.shape[1]}x{test_image.shape[0]})")
    else:
        print("[!] Note: Using synthetic portrait image for latency benchmarking.")

    print("\n[2] Running Biometric Experiments (Separated Gallery vs Held-Out Probes)...")
    runner = BiometricBenchmarkRunner(dimension=512, seed=42)
    report = runner.run_full_benchmark(
        face_engine=face_engine,
        test_image=test_image,
        n_enrolled=30,
        n_impostors=30,
        samples_per_subject=6,
    )

    print("\n" + "-" * 80)
    print("1. SYSTEM & MODEL CONFIGURATION")
    print("-" * 80)
    print(f"  Biometric Model Zoo:   {report.model_name} (SCRFD 2D Landmarks + ArcFace 512-d)")
    print(f"  Execution Providers:   {', '.join(report.execution_providers)}")
    print(f"  Operating System:      {report.system_info.get('os')} {report.system_info.get('release')}")
    print(f"  Processor / Arch:      {report.system_info.get('processor')}")
    print(f"  Python Runtime:        {report.system_info.get('python_version')}")

    print("\n" + "-" * 80)
    print("2. REAL HARDWARE LATENCY & FPS (Measured Directly on Host)")
    print("-" * 80)
    lat = report.latency
    lat_rows = [
        ["1. SCRFD Face Detection", f"{lat.detection_ms:.2f} ms", f"{1000.0 / max(1e-3, lat.detection_ms):.1f} FPS"],
        ["2. ArcFace Alignment & Embedding", f"{lat.embedding_ms:.2f} ms", f"{1000.0 / max(1e-3, lat.embedding_ms):.1f} FPS"],
        ["3. FAISS IndexFlatIP Search", f"{lat.search_ms:.2f} ms", f"{1000.0 / max(1e-3, lat.search_ms):.1f} FPS"],
        ["4. Anti-Spoofing / Liveness Check", f"{lat.liveness_ms:.2f} ms", f"{1000.0 / max(1e-3, lat.liveness_ms):.1f} FPS"],
        ["Total Pipeline Latency (End-to-End)", f"{lat.total_pipeline_ms:.2f} ms", f"{lat.effective_fps:.1f} FPS"],
    ]
    print(format_table(["Pipeline Stage", "Average Latency", "Throughput"], lat_rows))

    print("\n" + "-" * 80)
    print("3. VERIFICATION METRICS AT DEFAULT THRESHOLD (tau = 0.60)")
    print("-" * 80)
    def_m = report.metrics_at_default
    print(f"  Total Queries Evaluated:       {def_m.total_queries} ({def_m.genuine_queries} genuine, {def_m.impostor_queries} impostor)")
    print(f"  Recognition Accuracy:          {def_m.accuracy * 100:.2f}%")
    print(f"  Precision:                     {def_m.precision * 100:.2f}%")
    print(f"  Recall (True Positive Rate):   {def_m.recall * 100:.2f}%")
    print(f"  F1-Score:                      {def_m.f1_score:.4f}")
    print(f"  False Acceptance Rate (FAR):   {def_m.far * 100:.2f}% (False match count: {def_m.false_positives})")
    print(f"  False Rejection Rate (FRR):    {def_m.frr * 100:.2f}% (False reject count: {def_m.false_negatives})")
    print(f"  Unknown-Person Rejection:      {def_m.unknown_rejection_rate * 100:.2f}% (Correct unknown count: {def_m.true_negatives})")

    print("\n" + "-" * 80)
    print("4. SIMILARITY SCORE DISTRIBUTIONS")
    print("-" * 80)
    dist_rows = [
        [
            "Genuine Pairs (Same Subject)",
            f"{report.genuine_distribution.mean:.3f}",
            f"{report.genuine_distribution.std:.3f}",
            f"[{report.genuine_distribution.min:.3f}, {report.genuine_distribution.max:.3f}]",
            f"{report.genuine_distribution.p50:.3f}",
        ],
        [
            "Impostor Pairs (Different / Unknown)",
            f"{report.impostor_distribution.mean:.3f}",
            f"{report.impostor_distribution.std:.3f}",
            f"[{report.impostor_distribution.min:.3f}, {report.impostor_distribution.max:.3f}]",
            f"{report.impostor_distribution.p50:.3f}",
        ],
    ]
    print(format_table(["Category", "Mean Cosine", "Std Dev", "Min / Max Range", "Median (P50)"], dist_rows))

    print("\n" + "-" * 80)
    print("5. THRESHOLD SENSITIVITY SWEEP (FAR vs FRR)")
    print("-" * 80)
    sweep_rows = [
        [
            f"{pt.threshold:.2f}",
            f"{pt.far * 100:6.2f}%",
            f"{pt.frr * 100:6.2f}%",
            f"{pt.accuracy * 100:6.2f}%",
            f"{pt.f1_score:6.4f}",
            "<- Optimal F1" if pt.threshold == report.metrics_at_optimal.threshold else (
                "<- EER Operating Point" if pt.threshold == report.equal_error_rate_threshold else ""
            ),
        ]
        for pt in report.threshold_sweep
    ]
    print(format_table(["Threshold (tau)", "FAR (False Accept)", "FRR (False Reject)", "Accuracy", "F1-Score", "Notes"], sweep_rows))

    # Save to disk
    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    print("\n" + "=" * 80)
    print(f"[+] Evaluation Report Successfully Saved: {OUTPUT_REPORT_PATH.resolve()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
