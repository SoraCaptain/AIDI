# app/config.py
import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    统一配置管理。
    
    优先级（从低到高）：
    1. 类属性中的默认值（硬编码）
    2. .env 文件中的值
    3. 系统环境变量（export 或 shell 传入）
    """
    
    # ========== LLM / VLM 配置 ==========
    openai_api_key: str = "dummy"
    openai_base_url: str = "http://localhost:8003/v1"
    model_name: str = "Qwen/Qwen3.6-35B-A3B"
    
    # ========== 外部服务地址 ==========
    public_base_url: str = "http://localhost:8400"
    cv_server: str = "http://localhost:8200"
    gdino_server: str = "http://localhost:8210"
    vlm_server: str = "http://localhost:8002"
    vlm_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    
    # ========== Langfuse 可观测性 ==========
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://jp.cloud.langfuse.com"
    
    # ========== LangGraph 特殊配置 ==========
    langgraph_strict_msgpack: bool = True
    
    # ========== 🆕 检查点持久化配置（本节核心） ==========
    # 是否启用 PostgreSQL（默认关，保持与 SQLite 兼容）
    use_postgres_checkpointer: bool = False
    # PostgreSQL 连接串（默认指向本地标准端口）
    postgres_checkpointer_uri: str = "postgres://postgres:postgres@localhost:5432/aidi_checkpoints"
    
    # ========== 本地存储路径 ==========
    sqlite_checkpoint_path: str = "data/checkpoint/langgraph_checkpoints.sqlite3"
    memory_db_path: str = "data/memory/vision_memory.sqlite3"
    session_id: str = "api-gateway-session-001"

    # Pydantic 配置：自动加载 .env 文件，忽略未知字段
    model_config = SettingsConfigDict(
        env_file=".env",                # 加载 .env
        env_file_encoding="utf-8",
        extra="ignore",                 # 忽略 .env 中多余的字段，避免报错
        case_sensitive=False,           # 不区分大小写
    )


# 创建单例供全局导入
settings = Settings()