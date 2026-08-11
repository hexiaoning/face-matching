from __future__ import annotations

from dataclasses import dataclass

from face_match.config import MODEL_VERSION, AppPaths, AppSettings
from face_match.database import FaceDatabase
from face_match.enrollment import EnrollmentService
from face_match.gpu import GpuInfo, gpu_info
from face_match.model_manager import ModelManager
from face_match.vision.detector import ScrfdDetector
from face_match.vision.embedder import LvFaceEmbedder
from face_match.vision.matcher import MultiTemplateMatcher


@dataclass
class ApplicationServices:
    paths: AppPaths
    settings: AppSettings
    models: ModelManager
    database: FaceDatabase
    detector: ScrfdDetector
    embedder: LvFaceEmbedder
    matcher: MultiTemplateMatcher
    enrollment: EnrollmentService
    gpu: GpuInfo

    def refresh_matcher(self) -> None:
        self.matcher.refresh(self.database.load_embeddings(MODEL_VERSION))


def build_services(paths: AppPaths, settings: AppSettings) -> ApplicationServices:
    models = ModelManager(paths.models)
    database = FaceDatabase(paths.database)
    detector = ScrfdDetector(models.detector_path, input_size=settings.detector_size)
    embedder = LvFaceEmbedder(models.recognizer_path)
    matcher = MultiTemplateMatcher()
    matcher.refresh(database.load_embeddings(MODEL_VERSION))
    enrollment = EnrollmentService(database, detector, embedder, paths.photos)
    return ApplicationServices(
        paths=paths,
        settings=settings,
        models=models,
        database=database,
        detector=detector,
        embedder=embedder,
        matcher=matcher,
        enrollment=enrollment,
        gpu=gpu_info(),
    )
