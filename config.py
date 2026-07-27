from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    generator_model: str
    judge_model: str
    pass_threshold: float
    log_level: str


def get_settings() -> Settings:
    return Settings(
        anthropic_api_key=_get("ANTHROPIC_API_KEY"),
        generator_model=_get("GENERATOR_MODEL", "claude-sonnet-5"),
        judge_model=_get("JUDGE_MODEL", "claude-opus-4-8"),
        pass_threshold=float(_get("PASS_THRESHOLD", "3.5")),
        log_level=_get("LOG_LEVEL", "INFO"),
    )
