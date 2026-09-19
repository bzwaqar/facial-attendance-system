"""Custom exceptions for face processing and recognition pipeline."""


class FaceProcessingError(Exception):
    """Base exception for face processing errors."""

    def __init__(self, message: str = "An error occurred during face processing."):
        self.message = message
        super().__init__(self.message)


class InvalidImageError(FaceProcessingError):
    """Raised when an input image is invalid, corrupt, or unreadable."""

    def __init__(self, message: str = "Provided image is invalid, unreadable, or corrupt."):
        super().__init__(message)


class ModelNotLoadedError(FaceProcessingError):
    """Raised when the face recognition or detection model fails to initialize."""

    def __init__(self, message: str = "Face recognition model could not be loaded."):
        super().__init__(message)
