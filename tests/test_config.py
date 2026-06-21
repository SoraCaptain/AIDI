# test_config.py
from app.config import settings

print(f"use_postgres: {settings.use_postgres_checkpointer}")
print(f"postgres_uri: {settings.postgres_checkpointer_uri}")
print(f"vlm_server: {settings.vlm_server}")
print(f"model_name: {settings.model_name}")