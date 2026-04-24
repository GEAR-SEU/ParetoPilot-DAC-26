from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or Path(__file__).resolve().parents[1] / "config.yaml"

    @lru_cache(maxsize=1)
    def load(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}

        with open(self.config_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        if not isinstance(data, dict):
            return {}

        return data

    def get(self, *keys: str, default: Any = None) -> Any:
        value: Any = self.load()
        for key in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(key)
            if value is None:
                return default
        return value


CONFIG = ConfigLoader()


def load_config() -> dict[str, Any]:
    return CONFIG.load()


def get_config_value(*keys: str, default: Any = None) -> Any:
    return CONFIG.get(*keys, default=default)
