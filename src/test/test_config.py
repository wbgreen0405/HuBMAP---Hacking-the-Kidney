from src import config

def test_get_settings():
    s = config.get_settings()
    assert "BATCH_SIZE" in s
    assert s["IMAGE_SIZE"] > 0
