"""查询流程配置管理模块。

所有环境变量统一由 ``knowledge.core.config`` 读取。
"""

from dataclasses import dataclass
from typing import Optional

from knowledge.core.config import get_settings


@dataclass
class QueryConfig:
    """查询流程配置。"""

    # ==================== 文本处理配置 ====================
    max_context_chars: int = 12000

    # ==================== Rerank 配置 ====================
    rerank_max_top_k: int = 10
    rerank_min_top_k: int = 3
    rerank_gap_ratio: float = 0.25
    rerank_gap_abs: float = 0.08

    # ==================== RRF 配置 ====================
    rrf_k: int = 60
    rrf_max_results: int = 10

    # ==================== 检索配置 ====================
    embedding_search_limit: int = 5
    hyde_search_limit: int = 5

    # ==================== 商品确认节点配置 ====================
    item_name_high_confidence: float = 0.75
    item_name_mid_confidence: float = 0.45
    item_name_score_gap: float = 0.08
    item_name_max_options: int = 3
    item_name_dense_weight: float = 0.5
    item_name_sparse_weight: float = 0.5

    # ==================== LLM 配置 ====================
    openai_api_base: str = ""
    openai_api_key: str = ""
    default_model: str = ""
    item_model: str = ""

    # ==================== Milvus 配置 ====================
    milvus_url: str = ""
    chunks_collection: str = ""
    item_name_collection: str = ""
    entity_name_collection: str = ""

    # ==================== MCP 配置 ====================
    mcp_dashscope_base_url: str = ""

    @classmethod
    def from_env(cls) -> "QueryConfig":
        """从环境变量加载配置。

        Returns:
            配置实例。
        """
        settings = get_settings()
        return cls(
            max_context_chars=settings.max_context_chars,
            rerank_max_top_k=settings.rerank_max_top_k,
            rerank_min_top_k=settings.rerank_min_top_k,
            rerank_gap_ratio=settings.rerank_gap_ratio,
            rerank_gap_abs=settings.rerank_gap_abs,
            rrf_k=settings.rrf_k,
            rrf_max_results=settings.rrf_max_results,
            embedding_search_limit=settings.embedding_search_limit,
            hyde_search_limit=settings.hyde_search_limit,
            item_name_high_confidence=settings.item_name_high_confidence,
            item_name_mid_confidence=settings.item_name_mid_confidence,
            item_name_score_gap=settings.item_name_score_gap,
            item_name_max_options=settings.item_name_max_options,
            item_name_dense_weight=settings.item_name_dense_weight,
            item_name_sparse_weight=settings.item_name_sparse_weight,
            openai_api_base=settings.openai_api_base,
            openai_api_key=settings.openai_api_key,
            default_model=settings.llm_default_model,
            item_model=settings.item_model,
            milvus_url=settings.milvus_url,
            chunks_collection=settings.chunks_collection,
            item_name_collection=settings.item_name_collection,
            entity_name_collection=settings.entity_name_collection,
            mcp_dashscope_base_url=settings.mcp_dashscope_base_url,
        )


_config: Optional[QueryConfig] = None


def get_config() -> QueryConfig:
    """获取配置单例。

    Returns:
        全局配置实例。
    """
    global _config
    if _config is None:
        _config = QueryConfig.from_env()
    return _config
