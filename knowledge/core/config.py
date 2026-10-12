"""Application configuration.

Environment variables are loaded from ``knowledge/.env`` first for backward
compatibility and from the project root second. Existing OS environment
variables always take precedence.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_ROOT = PROJECT_ROOT / "knowledge"

for env_path in (KNOWLEDGE_ROOT / ".env", PROJECT_ROOT / ".env"):
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = _env(name)
        if value:
            return value
    return default


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    """Typed application settings."""

    project_root: Path
    knowledge_root: Path

    app_name: str
    app_env: str
    app_host: str
    app_port: int
    log_level: str

    openai_api_key: str
    openai_api_base: str
    llm_default_model: str
    llm_default_temperature: float
    vl_model: str
    item_model: str

    bge_m3_path: str
    bge_device: str
    bge_fp16: bool
    bge_reranker_large: str
    bge_reranker_device: str
    bge_reranker_fp16: bool
    embedding_dim: int

    mineru_api_token: str
    mineru_base_url: str
    mineru_model_source: str
    modelscope_cache: str
    hf_home: str
    md_root_dir: str

    milvus_url: str
    chunks_collection: str
    item_name_collection: str
    entity_name_collection: str
    milvus_metric_type: str
    milvus_min_cosine_score: float

    mongo_url: str
    mongo_db_name: str

    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket_name: str
    minio_secure: bool

    redis_url: str
    mcp_dashscope_base_url: str

    max_context_chars: int
    rerank_max_top_k: int
    rerank_min_top_k: int
    rerank_gap_ratio: float
    rerank_gap_abs: float
    rrf_k: int
    rrf_max_results: int
    embedding_search_limit: int
    hyde_search_limit: int

    item_name_high_confidence: float
    item_name_mid_confidence: float
    item_name_score_gap: float
    item_name_max_options: int
    item_name_dense_weight: float
    item_name_sparse_weight: float

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            project_root=PROJECT_ROOT,
            knowledge_root=KNOWLEDGE_ROOT,
            app_name=_env("APP_NAME", "shopkeeper-brain"),
            app_env=_env("APP_ENV", "development"),
            app_host=_env("APP_HOST", "0.0.0.0"),
            app_port=_env_int("APP_PORT", 8011),
            log_level=_env("LOG_LEVEL", "INFO").upper(),
            openai_api_key=_env_first("OPENAI_API_KEY", "OPEN_API_KEY"),
            openai_api_base=_env("OPENAI_API_BASE", "https://api.openai.com/v1"),
            llm_default_model=_env("LLM_DEFAULT_MODEL", "gpt-4o-mini"),
            llm_default_temperature=_env_float("LLM_DEFAULT_TEMPERATURE", 0.0),
            vl_model=_env("VL_MODEL", _env("VISION_MODEL", "gpt-4o-mini")),
            item_model=_env("ITEM_MODEL", "gpt-4o-mini"),
            bge_m3_path=_env("BGE_M3_PATH", "BAAI/bge-m3"),
            bge_device=_env("BGE_DEVICE", "cpu"),
            bge_fp16=_env_bool("BGE_FP16", False),
            bge_reranker_large=_env("BGE_RERANKER_LARGE", "BAAI/bge-reranker-large"),
            bge_reranker_device=_env("BGE_RERANKER_DEVICE", _env("BGE_DEVICE", "cpu")),
            bge_reranker_fp16=_env_bool(
                "BGE_RERANKER_FP16", _env_bool("BGE_FP16", False)
            ),
            embedding_dim=_env_int("EMBEDDING_DIM", 1024),
            mineru_api_token=_env("MINERU_API_TOKEN"),
            mineru_base_url=_env("MINERU_BASE_URL"),
            mineru_model_source=_env("MINERU_MODEL_SOURCE", "modelscope"),
            modelscope_cache=_env("MODELSCOPE_CACHE", ".cache/modelscope"),
            hf_home=_env("HF_HOME", ".cache/huggingface"),
            md_root_dir=_env("MD_ROOT_DIR", "knowledge/temp_data"),
            milvus_url=_env("MILVUS_URL", "http://localhost:19530"),
            chunks_collection=_env("CHUNKS_COLLECTION", "chunks"),
            item_name_collection=_env("ITEM_NAME_COLLECTION", "item_name"),
            entity_name_collection=_env("ENTITY_NAME_COLLECTION", "entity_name"),
            milvus_metric_type=_env("MILVUS_METRIC_TYPE", "COSINE"),
            milvus_min_cosine_score=_env_float("MILVUS_MIN_COSINE_SCORE", 0.3),
            mongo_url=_env("MONGO_URL", "mongodb://localhost:27017"),
            mongo_db_name=_env("MONGO_DB_NAME", "shopkeeper"),
            minio_endpoint=_env("MINIO_ENDPOINT", "localhost:9000"),
            minio_access_key=_env("MINIO_ACCESS_KEY", "minioadmin"),
            minio_secret_key=_env("MINIO_SECRET_KEY", "minioadmin"),
            minio_bucket_name=_env("MINIO_BUCKET_NAME", "shopkeeper"),
            minio_secure=_env_bool("MINIO_SECURE", False),
            redis_url=_env("REDIS_URL", "redis://localhost:6379/0"),
            mcp_dashscope_base_url=_env("MCP_DASHSCOPE_BASE_URL"),
            max_context_chars=_env_int("MAX_CONTEXT_CHARS", 12000),
            rerank_max_top_k=_env_int("RERANK_MAX_TOP_K", 10),
            rerank_min_top_k=_env_int("RERANK_MIN_TOP_K", 3),
            rerank_gap_ratio=_env_float("RERANK_GAP_RATIO", 0.25),
            rerank_gap_abs=_env_float("RERANK_GAP_ABS", 0.08),
            rrf_k=_env_int("RRF_K", 60),
            rrf_max_results=_env_int("RRF_MAX_RESULTS", 10),
            embedding_search_limit=_env_int("EMBEDDING_SEARCH_LIMIT", 5),
            hyde_search_limit=_env_int("HYDE_SEARCH_LIMIT", 5),
            item_name_high_confidence=_env_float("ITEM_NAME_HIGH_CONFIDENCE", 0.75),
            item_name_mid_confidence=_env_float("ITEM_NAME_MID_CONFIDENCE", 0.45),
            item_name_score_gap=_env_float("ITEM_NAME_SCORE_GAP", 0.08),
            item_name_max_options=_env_int("ITEM_NAME_MAX_OPTIONS", 3),
            item_name_dense_weight=_env_float("ITEM_NAME_DENSE_WEIGHT", 0.5),
            item_name_sparse_weight=_env_float("ITEM_NAME_SPARSE_WEIGHT", 0.5),
        )

    def safe_dict(self) -> Mapping[str, object]:
        """Return settings safe for logging."""

        secret_names = {
            "openai_api_key",
            "mineru_api_token",
            "minio_access_key",
            "minio_secret_key",
        }
        result: dict[str, object] = {}
        for name, value in asdict(self).items():
            result[name] = "***" if name in secret_names and value else value
        return result


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


def reset_settings_cache() -> None:
    """Clear the cached settings object. Mainly used by tests."""

    get_settings.cache_clear()
