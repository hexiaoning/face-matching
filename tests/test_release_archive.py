import zipfile

from scripts.create_release_archive import create_archive, sha256


def test_release_archive_contains_per_file_and_archive_checksums(tmp_path):
    source = tmp_path / "FaceMatching-test"
    internal = source / "_internal"
    internal.mkdir(parents=True)
    (source / "FaceMatching.exe").write_bytes(b"application")
    (internal / "model.onnx").write_bytes(b"model")
    archive = tmp_path / "FaceMatching-test.zip"

    assert create_archive(source, archive) == archive

    manifest = (source / "SHA256SUMS.txt").read_text(encoding="utf-8")
    assert sha256(source / "FaceMatching.exe") in manifest
    assert "_internal/model.onnx" in manifest
    assert archive.with_suffix(".zip.sha256").is_file()
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    assert "FaceMatching-test/FaceMatching.exe" in names
    assert "FaceMatching-test/SHA256SUMS.txt" in names
