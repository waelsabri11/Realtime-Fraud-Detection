"""Runtime configuration (environment variables / .env, no secrets in the repo)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

SECONDS_PER_DAY = 86_400


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- data (never copied into the repo) ---
    ieee_data_dir: Path = Path("~/Projects/ieee-fraud-detection").expanduser()
    work_dir: Path = Path("work")

    # --- infrastructure ---
    database_url: str = "postgresql+psycopg://fraud:fraud@localhost:5432/fraud"
    redis_url: str = "redis://localhost:6379/0"
    kafka_bootstrap: str = "localhost:19092"
    topic_transactions: str = "transactions"
    topic_labels: str = "labels"
    topic_partitions: int = 3
    consumer_group: str = "fraud-scorer"
    api_url: str = "http://localhost:8000"

    # --- model registry ---
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment: str = "fraud-detection"
    model_name: str = "fraud-lgbm"
    # registry alias ("champion") or an explicit version number ("3")
    model_version: str = "champion"
    model_reload_seconds: float = 10.0

    # --- time-based split, in days of TransactionDT (day = DT // 86400) ---
    train_end_day: int = 110
    replay_start_day: int = 146
    label_delay_days: float = 7.0
    seed: int = 42

    # --- alert allocation ---
    review_capacity_units: int = 100  # analyst review units per simulated day
    high_amount_threshold: float = 500.0  # cases above this need a senior review
    high_amount_cost: int = 2

    # --- online store ---
    feature_prefix: str = "fs"

    @property
    def label_delay_seconds(self) -> int:
        return int(self.label_delay_days * SECONDS_PER_DAY)

    @property
    def train_end_dt(self) -> int:
        return self.train_end_day * SECONDS_PER_DAY

    @property
    def replay_start_dt(self) -> int:
        return self.replay_start_day * SECONDS_PER_DAY

    @property
    def cache_dir(self) -> Path:
        return self.work_dir / "cache"

    @property
    def results_dir(self) -> Path:
        return self.work_dir / "results"

    api_timeout_seconds: float = Field(default=10.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
