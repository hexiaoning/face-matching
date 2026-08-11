from face_matching.security import LocalVault, mask_government_id


def test_vault_round_trip_and_digest(tmp_path):
    vault = LocalVault(tmp_path / "master.key")
    encrypted = vault.encrypt_text("110101199001011234")
    assert b"110101" not in encrypted
    assert vault.decrypt_text(encrypted) == "110101199001011234"
    assert vault.keyed_digest(" 110101199001011234 ") == vault.keyed_digest("110101199001011234")


def test_mask_government_id():
    assert mask_government_id("110101199001011234") == "110***********1234"
    assert mask_government_id("1234") == "****"
