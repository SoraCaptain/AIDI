# test_config.py
from app.config import settings
from utils.logger import logger

logger.info(f"use_postgres: {settings.use_postgres_checkpointer}")
logger.info(f"postgres_uri: {settings.postgres_checkpointer_uri}")
logger.info(f"vlm_server: {settings.vlm_server}")
logger.info(f"model_name: {settings.model_name}")