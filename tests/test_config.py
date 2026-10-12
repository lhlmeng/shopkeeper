from knowledge.core.config import get_settings, reset_settings_cache
from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.query_processor.config import QueryConfig


def test_project_paths_exist() -> None:
    settings = get_settings()

    assert settings.project_root.exists()
    assert settings.knowledge_root.exists()
    assert settings.knowledge_root.name == "knowledge"


def test_safe_dict_keeps_expected_keys_without_exposing_secret_names() -> None:
    settings = get_settings()
    safe = settings.safe_dict()

    assert "openai_api_key" in safe
    assert "minio_secret_key" in safe
    assert safe["openai_api_key"] in {"", "***"}
    assert safe["minio_secret_key"] in {"", "***"}


def test_query_config_uses_central_settings(monkeypatch) -> None:
    monkeypatch.setenv("MAX_CONTEXT_CHARS", "4321")
    reset_settings_cache()

    config = QueryConfig.from_env()

    assert config.max_context_chars == 4321


def test_import_config_uses_central_settings(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    reset_settings_cache()

    config = ImportConfig.from_env()

    assert config.embedding_dim == 768
