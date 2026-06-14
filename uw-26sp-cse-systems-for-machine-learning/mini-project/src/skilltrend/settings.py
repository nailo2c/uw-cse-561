from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = Field(default="sk-replace-me", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    model: str = Field(default="gpt-4o-mini", alias="SKILLTREND_MODEL")
    workers: int = Field(default=4, alias="SKILLTREND_WORKERS")
    max_postings_per_company: int = Field(default=50, alias="SKILLTREND_MAX_POSTINGS_PER_COMPANY")
    data_dir: Path = Field(default=Path("./data"), alias="SKILLTREND_DATA_DIR")
    fake_llm: bool = Field(default=False, alias="SKILLTREND_FAKE_LLM")
    # Client-side requests-per-minute cap. 0 = unlimited (use this on paid
    # tier where the backend queues for you). 10 = Gemini free-tier flash-lite.
    rpm: int = Field(default=0, alias="SKILLTREND_RPM")

    config_dir: Path = Path("./config")

    @property
    def postings_csv(self) -> Path:
        return self.data_dir / "postings" / "postings.csv"

    @property
    def extractions_csv(self) -> Path:
        return self.data_dir / "extractions" / "extractions.csv"

    @property
    def metrics_dir(self) -> Path:
        return self.data_dir / "metrics"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    def load_companies(self) -> dict[str, list[dict]]:
        path = self.config_dir / "companies.yaml"
        with path.open() as f:
            return yaml.safe_load(f) or {}

    def load_taxonomy(self) -> dict[str, list[str]]:
        path = self.config_dir / "taxonomy.yaml"
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        return data.get("canonical", {})


settings = Settings()
