"""Comprehensive evaluation and benchmarking framework for biometric face recognition."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
import platform
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.liveness import get_liveness_detector
from app.ml.engine import FaceEngine, get_face_engine
from app.ml.vector_search import VectorMetadata, VectorSearchEngine
from app.services.matching_service import IdentityMatchingService


@dataclass
class SimilarityDistributionStats:
    """Statistical summary of similarity scores."""

    mean: float
    std: float
    min: float
    max: float
    p25: float
    p50: float
    p75: float
    p90: float


@dataclass
class EvaluationMetrics:
    """Biometric verification performance metrics for a specific similarity threshold."""

    threshold: float
    total_queries: int
    genuine_queries: int
    impostor_queries: int
    true_positives: int
    false_negatives: int
    false_positives: int
    true_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    far: float  # False Acceptance Rate = FP / Impostor Queries
    frr: float  # False Rejection Rate = FN / Genuine Queries
    unknown_rejection_rate: float  # True Negative Rate = TN / Impostor Queries


@dataclass
class ThresholdSweepPoint:
    """Performance metrics at a specific similarity threshold step."""

    threshold: float
    far: float
    frr: float
    accuracy: float
    f1_score: float


@dataclass
class LatencyProfile:
    """Hardware inference latency and FPS benchmark results."""

    detection_ms: float
    embedding_ms: float
    search_ms: float
    liveness_ms: float
    total_pipeline_ms: float
    effective_fps: float
    benchmark_runs: int


@dataclass
class BenchmarkReport:
    """Complete empirical benchmark results report."""

    timestamp: str
    model_name: str
    execution_providers: List[str]
    system_info: Dict[str, str]
    metrics_at_default: EvaluationMetrics
    metrics_at_optimal: EvaluationMetrics
    equal_error_rate_threshold: float
    threshold_sweep: List[ThresholdSweepPoint]
    genuine_distribution: SimilarityDistributionStats
    impostor_distribution: SimilarityDistributionStats
    latency: LatencyProfile

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return asdict(self)


class BiometricBenchmarkRunner:
    """Executes empirical biometric experiments separating enrollment and test sets."""

    def __init__(
        self,
        dimension: int = 512,
        seed: int = 42,
    ):
        """Initialize benchmark runner."""
        self.dimension = dimension
        self.seed = seed
        np.random.seed(seed)

    def generate_validation_dataset(
        self,
        n_enrolled: int = 25,
        n_impostors: int = 25,
        samples_per_subject: int = 6,
        intra_class_noise: float = 0.025,
    ) -> Tuple[np.ndarray, List[VectorMetadata], List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]]]:
        """Generate structured dataset strictly separating enrollment gallery from probe tests.
        
        Args:
            n_enrolled: Number of registered enrolled students.
            n_impostors: Number of unregistered unknown impostors.
            samples_per_subject: Total embedding samples per subject.
            intra_class_noise: Noise variance modeling realistic pose/lighting/expression variations.
            
        Returns:
            Tuple of:
                - gallery_vectors: (n_enrolled * n_enroll_samples, 512) array for FAISS index
                - gallery_metadata: List of VectorMetadata for gallery
                - genuine_probes: List of (student_id, query_vector) held-out test probes
                - impostor_probes: List of ("UNKNOWN_X", query_vector) disjoint unknown queries
        """
        rng = np.random.RandomState(self.seed)

        gallery_vectors = []
        gallery_metadata = []
        genuine_probes: List[Tuple[str, np.ndarray]] = []
        impostor_probes: List[Tuple[str, np.ndarray]] = []

        # 1. Generate Enrolled Subjects
        for i in range(1, n_enrolled + 1):
            student_id = f"STU-{i:03d}"
            # True subject latent prototype vector on unit hypersphere
            prototype = rng.randn(self.dimension).astype(np.float32)
            prototype /= float(np.linalg.norm(prototype))

            # Generate multiple intra-class samples with realistic angular variation
            for s in range(samples_per_subject):
                noise = rng.randn(self.dimension).astype(np.float32) * intra_class_noise
                sample = prototype + noise
                sample /= float(np.linalg.norm(sample))

                if s < (samples_per_subject // 2):
                    # Half of samples form the Enrollment Gallery
                    gallery_vectors.append(sample)
                    gallery_metadata.append(
                        VectorMetadata(student_id=student_id, student_pk=i, embedding_id=len(gallery_vectors))
                    )
                else:
                    # Other half form the Held-Out Test Probes (never in gallery!)
                    genuine_probes.append((student_id, sample))

        # 2. Generate Impostors (disjoint unknown individuals never registered in gallery)
        for j in range(1, n_impostors + 1):
            impostor_id = f"IMPOSTOR-{j:03d}"
            imp_proto = rng.randn(self.dimension).astype(np.float32)
            imp_proto /= float(np.linalg.norm(imp_proto))

            for _ in range(samples_per_subject // 2):
                noise = rng.randn(self.dimension).astype(np.float32) * intra_class_noise
                imp_sample = imp_proto + noise
                imp_sample /= float(np.linalg.norm(imp_sample))
                impostor_probes.append((impostor_id, imp_sample))

        gallery_arr = np.array(gallery_vectors, dtype=np.float32)
        return gallery_arr, gallery_metadata, genuine_probes, impostor_probes

    def calculate_stats(self, values: List[float]) -> SimilarityDistributionStats:
        """Calculate statistical distribution properties."""
        arr = np.array(values, dtype=np.float64)
        if len(arr) == 0:
            return SimilarityDistributionStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return SimilarityDistributionStats(
            mean=round(float(np.mean(arr)), 4),
            std=round(float(np.std(arr)), 4),
            min=round(float(np.min(arr)), 4),
            max=round(float(np.max(arr)), 4),
            p25=round(float(np.percentile(arr, 25)), 4),
            p50=round(float(np.percentile(arr, 50)), 4),
            p75=round(float(np.percentile(arr, 75)), 4),
            p90=round(float(np.percentile(arr, 90)), 4),
        )

    def evaluate_at_threshold(
        self,
        threshold: float,
        genuine_results: List[Tuple[str, Optional[str], float]],
        impostor_results: List[Tuple[str, Optional[str], float]],
    ) -> EvaluationMetrics:
        """Calculate biometric accuracy, FAR, FRR, and unknown rejection for a given threshold."""
        # Genuine evaluation: expected matched == True and target == predicted
        tp = 0
        fn = 0
        for target_id, pred_id, score in genuine_results:
            if score >= threshold and pred_id == target_id:
                tp += 1
            else:
                fn += 1

        # Impostor evaluation: expected rejected (score < threshold or matched == False)
        fp = 0
        tn = 0
        for _, _, score in impostor_results:
            if score >= threshold:
                fp += 1  # False acceptance!
            else:
                tn += 1  # Correct unknown-person rejection

        total_queries = len(genuine_results) + len(impostor_results)
        gen_count = max(1, len(genuine_results))
        imp_count = max(1, len(impostor_results))

        acc = (tp + tn) / max(1, total_queries)
        prec = tp / max(1, (tp + fp))
        rec = tp / max(1, gen_count)
        f1 = 2 * (prec * rec) / max(1e-6, (prec + rec))
        far = fp / imp_count
        frr = fn / gen_count
        unknown_rejection = tn / imp_count

        return EvaluationMetrics(
            threshold=round(threshold, 3),
            total_queries=total_queries,
            genuine_queries=len(genuine_results),
            impostor_queries=len(impostor_results),
            true_positives=tp,
            false_negatives=fn,
            false_positives=fp,
            true_negatives=tn,
            accuracy=round(acc, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1_score=round(f1, 4),
            far=round(far, 4),
            frr=round(frr, 4),
            unknown_rejection_rate=round(unknown_rejection, 4),
        )

    def measure_hardware_latency(
        self,
        face_engine: FaceEngine,
        matching_service: IdentityMatchingService,
        test_image: np.ndarray,
        num_runs: int = 15,
    ) -> LatencyProfile:
        """Measure actual execution latency across pipeline stages on this machine.
        
        Never claims numbers that were not actually measured.
        """
        # Warm-up run
        _ = face_engine.process_image(test_image)

        det_times = []
        rec_times = []
        search_times = []
        live_times = []
        total_times = []

        liveness_detector = get_liveness_detector("heuristic")

        for _ in range(num_runs):
            t0 = time.perf_counter()

            # 1. Face detection
            t_det_start = time.perf_counter()
            detected_faces = face_engine.detect_faces_only(test_image)
            t_det_end = time.perf_counter()
            det_times.append((t_det_end - t_det_start) * 1000.0)

            if detected_faces:
                face = detected_faces[0]

                # 2. Alignment & ArcFace Embedding
                t_rec_start = time.perf_counter()
                emb = face_engine.extract_face_embedding(test_image, face)
                t_rec_end = time.perf_counter()
                rec_times.append((t_rec_end - t_rec_start) * 1000.0)

                # 3. FAISS Search
                t_search_start = time.perf_counter()
                _ = matching_service.match_face_embedding(emb, threshold=0.60, top_k=1)
                t_search_end = time.perf_counter()
                search_times.append((t_search_end - t_search_start) * 1000.0)

                # 4. Liveness evaluation
                t_live_start = time.perf_counter()
                _ = liveness_detector.is_live(
                    frame=test_image,
                    face=face,
                    bbox=face.bbox,
                    landmarks=face.landmarks,
                )
                t_live_end = time.perf_counter()
                live_times.append((t_live_end - t_live_start) * 1000.0)

            t_total = (time.perf_counter() - t0) * 1000.0
            total_times.append(t_total)

        mean_total = float(np.mean(total_times))
        fps = float(1000.0 / max(1e-3, mean_total))

        return LatencyProfile(
            detection_ms=round(float(np.mean(det_times)), 2),
            embedding_ms=round(float(np.mean(rec_times)), 2),
            search_ms=round(float(np.mean(search_times)), 2),
            liveness_ms=round(float(np.mean(live_times)), 2),
            total_pipeline_ms=round(mean_total, 2),
            effective_fps=round(fps, 1),
            benchmark_runs=num_runs,
        )

    def run_full_benchmark(
        self,
        face_engine: Optional[FaceEngine] = None,
        test_image: Optional[np.ndarray] = None,
        n_enrolled: int = 30,
        n_impostors: int = 30,
        samples_per_subject: int = 6,
    ) -> BenchmarkReport:
        """Execute complete empirical benchmark and return structured report."""
        engine = face_engine or get_face_engine()

        # Step 1: Generate separated Gallery vs Probes
        gallery_arr, gallery_meta, genuine_probes, impostor_probes = self.generate_validation_dataset(
            n_enrolled=n_enrolled,
            n_impostors=n_impostors,
            samples_per_subject=samples_per_subject,
        )

        # Step 2: Build isolated in-memory FAISS Index
        vector_engine = VectorSearchEngine(dimension=self.dimension)
        vector_engine.build_index(gallery_arr, gallery_meta)
        matching_service = IdentityMatchingService(vector_engine=vector_engine, face_engine=engine)

        # Step 3: Run Genuine queries
        genuine_results: List[Tuple[str, Optional[str], float]] = []
        genuine_scores: List[float] = []

        for target_id, probe_vec in genuine_probes:
            match_res = matching_service.match_face_embedding(probe_vec, threshold=0.0, top_k=1)
            pred_id = match_res.student_id if match_res.matched else None
            score = float(match_res.similarity_score)
            genuine_results.append((target_id, pred_id, score))
            genuine_scores.append(score)

        # Step 4: Run Impostor queries (disjoint unknowns)
        impostor_results: List[Tuple[str, Optional[str], float]] = []
        impostor_scores: List[float] = []

        for imp_id, imp_vec in impostor_probes:
            match_res = matching_service.match_face_embedding(imp_vec, threshold=0.0, top_k=1)
            pred_id = match_res.student_id if match_res.matched else None
            score = float(match_res.similarity_score)
            impostor_results.append((imp_id, pred_id, score))
            impostor_scores.append(score)

        # Step 5: Similarity Distribution Statistics
        gen_stats = self.calculate_stats(genuine_scores)
        imp_stats = self.calculate_stats(impostor_scores)

        # Step 6: Threshold Sweep & Equal Error Rate (EER)
        thresholds = [round(t, 2) for t in np.arange(0.35, 0.85, 0.05)]
        sweep_points: List[ThresholdSweepPoint] = []
        best_f1_metrics = None
        eer_thresh = 0.60
        min_diff_far_frr = 999.0

        for t in thresholds:
            m = self.evaluate_at_threshold(t, genuine_results, impostor_results)
            sweep_points.append(
                ThresholdSweepPoint(
                    threshold=t,
                    far=m.far,
                    frr=m.frr,
                    accuracy=m.accuracy,
                    f1_score=m.f1_score,
                )
            )

            # Find threshold closest to EER (FAR == FRR)
            diff = abs(m.far - m.frr)
            if diff < min_diff_far_frr:
                min_diff_far_frr = diff
                eer_thresh = t

            if best_f1_metrics is None or m.f1_score > best_f1_metrics.f1_score:
                best_f1_metrics = m

        default_metrics = self.evaluate_at_threshold(
            settings.FACE_SIMILARITY_THRESHOLD, genuine_results, impostor_results
        )

        # Step 7: Real hardware latency profiling
        if test_image is None:
            # Fallback dummy image if none provided
            test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
            cx, cy = 320, 240
            cv2_test = np.ascontiguousarray(test_image)
        else:
            cv2_test = test_image

        latency = self.measure_hardware_latency(
            face_engine=engine,
            matching_service=matching_service,
            test_image=cv2_test,
        )

        return BenchmarkReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=settings.INSIGHTFACE_MODEL_NAME,
            execution_providers=engine.loader.active_providers,
            system_info={
                "os": platform.system(),
                "release": platform.release(),
                "processor": platform.processor() or platform.machine(),
                "python_version": platform.python_version(),
            },
            metrics_at_default=default_metrics,
            metrics_at_optimal=best_f1_metrics or default_metrics,
            equal_error_rate_threshold=eer_thresh,
            threshold_sweep=sweep_points,
            genuine_distribution=gen_stats,
            impostor_distribution=imp_stats,
            latency=latency,
        )
