class FaceMatchingError(RuntimeError):
    """Base exception for a user-actionable application error."""


class GPUUnavailableError(FaceMatchingError):
    """Raised when CUDA or OpenVINO GPU inference cannot be guaranteed."""


class ModelMissingError(FaceMatchingError):
    """Raised when required ONNX weights are absent."""


class EnrollmentError(FaceMatchingError):
    """Raised when an enrollment photo cannot safely be accepted."""
