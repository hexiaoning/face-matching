from face_matching.privacy import mask_id_card, redact_source_credentials


def test_sensitive_values_are_masked() -> None:
    assert mask_id_card("110101199001011234") == "1101**********1234"
    assert mask_id_card("ABCD") == "****"


def test_camera_credentials_are_removed() -> None:
    source = "rtsp://admin:secret@192.0.2.1/live"
    safe = redact_source_credentials(source)
    assert safe == "rtsp://192.0.2.1/live"
    assert "secret" not in safe
