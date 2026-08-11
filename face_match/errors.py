class FaceMatchError(RuntimeError):
    """Base class for errors that can be shown directly to an operator."""


class GpuUnavailableError(FaceMatchError):
    """Raised when CUDA inference cannot be guaranteed."""


class ModelDownloadError(FaceMatchError):
    """Raised when a model is missing, corrupt, or cannot be downloaded."""


class EnrollmentError(FaceMatchError):
    """Raised when a reference photo cannot be enrolled safely."""


class VideoSourceError(FaceMatchError):
    """Raised when a video source cannot be opened or read."""
