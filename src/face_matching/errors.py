class FaceMatchingError(RuntimeError):
    """Base application error that is safe to show to a user."""


class GPUUnavailableError(FaceMatchingError):
    """Raised when CUDA/OpenVINO GPU inference cannot be guaranteed."""


class ModelMissingError(FaceMatchingError):
    """Raised when a required model file is absent or corrupt."""


class EnrollmentError(FaceMatchingError):
    """Raised when enrollment photos cannot produce usable templates."""
