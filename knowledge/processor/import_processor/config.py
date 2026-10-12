"""导入流程配置管理模块。

所有环境变量统一由 ``knowledge.core.config`` 读取。
"""

from dataclasses import dataclass, field
from typing import Optional, Set

from knowledge.core.config import get_settings


@dataclass
class ImportConfig:
    """导入流程配置"""

    # ==================== 文档处理配置 ====================
    max_content_length: int = 2000  # 切片最大长度
    img_content_length: int = 200  # 图片上下文最大长度
    min_content_length: int = 500  # 合并短内容的最小长度
    overlap_sentences: int = 1  # 句子级切分时的重叠句数
    item_name_chunk_k: int = 3  # 商品名识别时使用的切片数量
    item_name_chunk_size: int = 2500  # 商品名识别时使用的切片内容长度

    image_extensions: Set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    )

    # ==================== LLM 配置 ====================
    openai_api_base: str = ""
    openai_api_key: str = ""
    vl_model: str = ""
    item_model: str = ""
    default_model: str = ""

    mineru_api_token: str = ""
    mineru_base_url: str = ""

    # ==================== Milvus 配置 ====================
    milvus_url: str = ""
    chunks_collection: str = ""
    item_name_collection: str = ""
    entity_name_collection: str = ""


    # ==================== MinIO 配置 ====================
    minio_endpoint: str = ""
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = ""
    minio_secure: bool = False

    # ==================== 向量配置 ====================
    embedding_dim: int = 1024
    embedding_batch_size: int = 8

    # ==================== 速率限制 ====================
    requests_per_minute: int = 15  # 图片总结 API 速率限制

    @classmethod
    def from_env(cls) -> "ImportConfig":
        """从环境变量加载配置"""
        settings = get_settings()
        return cls(
            openai_api_base=settings.openai_api_base,
            openai_api_key=settings.openai_api_key,
            vl_model=settings.vl_model,
            item_model=settings.item_model,
            default_model=settings.llm_default_model,
            mineru_api_token=settings.mineru_api_token,
            mineru_base_url=settings.mineru_base_url,
            milvus_url=settings.milvus_url,
            chunks_collection=settings.chunks_collection,
            item_name_collection=settings.item_name_collection,
            entity_name_collection=settings.entity_name_collection,
            minio_endpoint=settings.minio_endpoint,
            minio_access_key=settings.minio_access_key,
            minio_secret_key=settings.minio_secret_key,
            minio_bucket=settings.minio_bucket_name,
            embedding_dim=settings.embedding_dim,
        )

    # http://192.168.200.130:9000/
    def get_minio_base_url(self):
        base_protocol = "https://" if self.minio_secure else "http://"
        return base_protocol + f"{self.minio_endpoint}"


# ==================== 全局单例 ====================
_config: Optional[ImportConfig] = None


def get_config() -> ImportConfig:
    """获取配置单例"""
    global _config
    if _config is None:
        _config = ImportConfig.from_env()
    return _config
