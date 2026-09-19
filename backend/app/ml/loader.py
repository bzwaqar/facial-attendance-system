"""Pretrained InsightFace model loader and singleton management."""

import logging
import threading
from typing import List, Optional

import onnxruntime
from insightface.app import FaceAnalysis

from app.core.config import settings
from app.ml.exceptions import ModelNotLoadedError

logger = logging.getLogger(__name__)


class InsightFaceModelLoader:
    """Thread-safe singleton manager for loading and reusing pretrained InsightFace models."""

    _instance: Optional["InsightFaceModelLoader"] = None
    _lock: threading.Lock = threading.Lock()
    _model: Optional[FaceAnalysis] = None
    _active_providers: List[str] = []

    def __new__(cls) -> "InsightFaceModelLoader":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(InsightFaceModelLoader, cls).__new__(cls)
        return cls._instance

    def _determine_execution_providers(self) -> List[str]:
        """Detect and select available execution providers (GPU vs CPU)."""
        available = onnxruntime.get_available_providers()
        logger.info("ONNX Runtime available execution providers: %s", available)

        if settings.USE_GPU and "CUDAExecutionProvider" in available:
            logger.info("Enabling GPU acceleration (CUDAExecutionProvider).")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        logger.info("Running on CPU execution provider (CPUExecutionProvider).")
        return ["CPUExecutionProvider"]

    def load_model(
        self,
        model_name: Optional[str] = None,
        det_size: Optional[int] = None,
        force_reload: bool = False,
    ) -> FaceAnalysis:
        """Load the pretrained InsightFace model once and cache it.
        
        Args:
            model_name: Name of model zoo package (e.g. 'buffalo_sc', 'buffalo_l').
            det_size: Input detection resolution square size (e.g. 640).
            force_reload: If True, forces reloading of the model.
            
        Returns:
            Configured and prepared FaceAnalysis instance.
        """
        target_model = model_name or settings.INSIGHTFACE_MODEL_NAME
        target_size = det_size or settings.DETECTION_INPUT_SIZE

        with self._lock:
            if self._model is not None and not force_reload:
                return self._model

            try:
                providers = self._determine_execution_providers()
                logger.info(
                    "Initializing FaceAnalysis model '%s' with providers %s...",
                    target_model,
                    providers,
                )

                # Initialize model loading only detection and recognition modules
                app = FaceAnalysis(
                    name=target_model,
                    providers=providers,
                    allowed_modules=["detection", "recognition"],
                )

                # ctx_id: 0 for GPU, -1 / 0 for CPU
                ctx_id = 0 if "CUDAExecutionProvider" in providers else 0
                app.prepare(ctx_id=ctx_id, det_size=(target_size, target_size))

                self._model = app
                self._active_providers = providers
                logger.info("FaceAnalysis model '%s' loaded and prepared successfully.", target_model)
                return self._model

            except Exception as exc:
                logger.error("Failed to load InsightFace model: %s", exc, exc_info=True)
                raise ModelNotLoadedError(f"Could not load InsightFace model '{target_model}': {exc}") from exc

    def get_model(self) -> FaceAnalysis:
        """Get or initialize the cached FaceAnalysis model instance."""
        if self._model is None:
            return self.load_model()
        return self._model

    @property
    def is_loaded(self) -> bool:
        """True if the model has been loaded and initialized in memory."""
        return self._model is not None

    @property
    def active_providers(self) -> List[str]:
        """List of active ONNX execution providers."""
        return self._active_providers


def get_model_loader() -> InsightFaceModelLoader:
    """Return the global singleton model loader instance."""
    return InsightFaceModelLoader()
