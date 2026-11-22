"""Configuración propia del pipeline; no descubre archivos .env externos."""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")
    comercial_mongodb_uri: str | None = None
    comercial_mongodb_db: str | None = None
    comercial_data_dir: str = str(Path.cwd() / "data/m5")
    comercial_ventanas: int = 2
    mongo_timeout_ms: int = 3000

    def comercial_dir(self) -> Path:
        return Path(self.comercial_data_dir)

@lru_cache
def get_settings() -> Settings:
    return Settings()
