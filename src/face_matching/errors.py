class FaceMatchingError(RuntimeError):
    """Base error shown to the user without a traceback."""


class GpuUnavailableError(FaceMatchingError):
    """Raised when CUDA inference cannot be guaranteed."""


class ModelError(FaceMatchingError):
    """Raised when model files are absent or incompatible."""


class EnrollmentError(FaceMatchingError):
    """Raised when enrollment photos cannot produce usable faces."""
