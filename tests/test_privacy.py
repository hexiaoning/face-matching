from face_matching.privacy import mask_id_card, redact_source_credentials


def test_identifiers_are_masked_even_when_short():
    assert mask_id_card("") == ""
    assert mask_id_card("A") == "*"
    assert mask_id_card("ABCD") == "****"
    assert mask_id_card("ABCDEF") == "A****F"
    assert mask_id_card("123456789012345678") == "1234**********5678"


def test_camera_credentials_are_not_persisted():
    assert (
        redact_source_credentials("rtsp://admin:secret@192.0.2.1/live")
        == "rtsp://192.0.2.1/live"
    )
    assert redact_source_credentials("RTMP://user@camera/live") == "RTMP://camera/live"
    assert redact_source_credentials(r"C:\videos\example.mp4") == r"C:\videos\example.mp4"
