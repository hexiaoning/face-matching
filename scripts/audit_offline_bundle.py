"""Require all application features, verified models and offline runtimes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PyInstaller.archive.readers import CArchiveReader
from face_matching.models import PROFILES, required_paths, required_person_vehicle_model


def audit(root: Path, profiles: list[str]) -> dict:
    model_root = root / '_internal/models'
    models = {required_person_vehicle_model(model_root)}
    for profile in profiles:
        models.update(required_paths(profile, model_root, verify_hash=True))
    if {path.name for path in model_root.iterdir()} != {model.name for model in models}:
        raise RuntimeError('Model payload does not match the selected full application profiles')
    modules = CArchiveReader(str(root / 'FaceMatching.exe')).open_embedded_archive('PYZ.pyz').toc
    required_modules = {
        'face_matching.__main__', 'face_matching.engine', 'face_matching.database',
        'face_matching.enrollment', 'face_matching.matching', 'face_matching.gui.main_window',
        'face_matching.gui.video_detection', 'face_matching.gui.image_tools',
        'face_matching.gui.visual_search', 'face_matching.gui.portrait_libraries',
        'face_matching.identity_filters', 'face_matching.visual_search', 'face_matching.runtime_check',
        'face_matching.annotation_export', 'face_matching.gui.annotation_compare',
    }
    if missing := sorted(required_modules - modules.keys()):
        raise RuntimeError(f'Missing application features: {missing}')
    dlls = {path.name.lower() for path in (root / '_internal').rglob('*.dll')}
    required_dlls = {'qwindows.dll', 'qt6widgets.dll', 'onnxruntime_providers_cuda.dll',
                     'openvino_intel_gpu_plugin.dll', 'cublas64_12.dll', 'cublaslt64_12.dll',
                     'cudart64_12.dll', 'cudnn64_9.dll'}
    if missing := sorted(required_dlls - dlls):
        raise RuntimeError(f'Missing offline runtime DLLs: {missing}')
    if 'icuuc.dll' in dlls:
        raise RuntimeError('External ICU shim must not shadow the Windows system DLL')
    resources = root / '_internal/face_matching/resources'
    for name in ('regions-2023.json', 'regions-source.json', 'REGIONS-LICENSE.txt'):
        if not (resources / name).is_file():
            raise RuntimeError(f'Missing offline region data: {name}')
    return {'ok': True, 'models': sorted(model.name for model in models),
            'profiles': profiles, 'required_modules': sorted(required_modules), 'dll_count': len(dlls)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--profiles', nargs='+', choices=sorted(PROFILES), required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory, args.profiles), indent=2))
