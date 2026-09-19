"""Unit tests for the biometric evaluation framework and metrics calculation."""

import numpy as np
import pytest

from app.evaluation.benchmark import (
    BenchmarkReport,
    BiometricBenchmarkRunner,
    EvaluationMetrics,
    SimilarityDistributionStats,
)


def test_validation_dataset_separation():
    """Verify evaluation dataset strictly separates gallery enrollment from probe test sets."""
    runner = BiometricBenchmarkRunner(dimension=512, seed=123)
    gallery_arr, gallery_meta, genuine_probes, impostor_probes = runner.generate_validation_dataset(
        n_enrolled=10,
        n_impostors=10,
        samples_per_subject=6,
    )

    # 10 subjects * 3 enrollment samples = 30 gallery vectors
    assert gallery_arr.shape == (30, 512)
    assert len(gallery_meta) == 30

    # 10 subjects * 3 test samples = 30 genuine probes
    assert len(genuine_probes) == 30

    # 10 impostor subjects * 3 samples = 30 impostor probes
    assert len(impostor_probes) == 30

    # Ensure all vectors are L2 unit-normalized
    norms = np.linalg.norm(gallery_arr, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    # Ensure impostors are strictly disjoint from enrolled IDs
    gallery_ids = {m.student_id for m in gallery_meta}
    impostor_ids = {imp[0] for imp in impostor_probes}
    assert gallery_ids.isdisjoint(impostor_ids)


def test_evaluation_metrics_calculation_perfect_match():
    """Verify metrics calculation when all genuine matches pass and impostors are rejected."""
    runner = BiometricBenchmarkRunner()

    genuine_results = [
        ("STU-001", "STU-001", 0.85),
        ("STU-002", "STU-002", 0.90),
    ]
    impostor_results = [
        ("IMP-001", "STU-001", 0.30),
        ("IMP-002", None, 0.25),
    ]

    metrics = runner.evaluate_at_threshold(
        threshold=0.60,
        genuine_results=genuine_results,
        impostor_results=impostor_results,
    )

    assert metrics.accuracy == 1.0
    assert metrics.far == 0.0
    assert metrics.frr == 0.0
    assert metrics.unknown_rejection_rate == 1.0
    assert metrics.true_positives == 2
    assert metrics.true_negatives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0


def test_evaluation_metrics_far_and_frr():
    """Verify false acceptance and false rejection rates are correctly computed."""
    runner = BiometricBenchmarkRunner()

    # 1 true positive, 1 false rejection (score < threshold)
    genuine_results = [
        ("STU-001", "STU-001", 0.85),
        ("STU-002", "STU-002", 0.45),  # Below threshold 0.60
    ]
    # 1 true negative, 1 false acceptance (score >= threshold)
    impostor_results = [
        ("IMP-001", "STU-001", 0.70),  # Impostor falsely accepted!
        ("IMP-002", None, 0.20),
    ]

    metrics = runner.evaluate_at_threshold(
        threshold=0.60,
        genuine_results=genuine_results,
        impostor_results=impostor_results,
    )

    assert metrics.true_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.false_positives == 1
    assert metrics.true_negatives == 1

    assert metrics.far == 0.50
    assert metrics.frr == 0.50
    assert metrics.accuracy == 0.50


def test_calculate_similarity_distribution_stats():
    """Verify statistics calculation on similarity arrays."""
    runner = BiometricBenchmarkRunner()
    scores = [0.10, 0.20, 0.30, 0.40, 0.50]

    stats = runner.calculate_stats(scores)
    assert stats.min == 0.10
    assert stats.max == 0.50
    assert stats.mean == pytest.approx(0.30, abs=1e-4)
    assert stats.p50 == pytest.approx(0.30, abs=1e-4)


def test_calculate_stats_empty():
    """Verify empty array handled safely."""
    runner = BiometricBenchmarkRunner()
    stats = runner.calculate_stats([])
    assert stats.mean == 0.0
    assert stats.max == 0.0
