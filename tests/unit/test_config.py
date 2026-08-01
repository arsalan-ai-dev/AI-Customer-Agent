from app.config.settings import settings

def test_settings_load_successfully():
    """Verify core application settings loaded correctly."""
    assert settings.APP_NAME is not None
    assert settings.VERSION == "2.0.0"
    assert settings.RETRIEVAL_TOP_K == 3
    assert settings.GROQ_MODEL_NAME == "llama-3.1-8b-instant"