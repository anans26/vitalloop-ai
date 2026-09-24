"""The promotion rules artifact: `configs/promotion-v*.yaml`.

Loaded and validated the way `loop/gate/criteria.py` loads the gate criteria:
unknown keys and missing keys are errors, because a rules file that silently
ignored a misspelt threshold would promote on a bar nobody set.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from ml.config import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_RULES_VERSION = "promotion-v1"

_REQUIRED = ("version", "min_shadow_requests", "min_reason_length")


class PromotionRulesError(ValueError):
    """The rules file is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class PromotionRules:
    version: str
    min_shadow_requests: int
    min_reason_length: int

    def to_record(self) -> dict:
        return {
            "version": self.version,
            "min_shadow_requests": self.min_shadow_requests,
            "min_reason_length": self.min_reason_length,
        }


def rules_path(version: str, config_dir: Path = CONFIG_DIR) -> Path:
    return Path(config_dir) / f"{version}.yaml"


def load_rules(
    version: str = DEFAULT_RULES_VERSION, config_dir: Path = CONFIG_DIR
) -> PromotionRules:
    path = rules_path(version, config_dir)
    if not path.exists():
        raise PromotionRulesError(f"no promotion rules file for {version!r}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise PromotionRulesError(f"{path.name} is not a mapping")

    missing = [key for key in _REQUIRED if key not in raw]
    unknown = sorted(set(raw) - set(_REQUIRED))
    if missing or unknown:
        raise PromotionRulesError(f"{path.name}: missing {missing}, unknown {unknown}")
    if raw["version"] != version:
        raise PromotionRulesError(f"{path.name} declares version {raw['version']!r}")

    try:
        rules = PromotionRules(
            version=str(raw["version"]),
            min_shadow_requests=int(raw["min_shadow_requests"]),
            min_reason_length=int(raw["min_reason_length"]),
        )
    except (TypeError, ValueError) as error:
        raise PromotionRulesError(f"{path.name}: {error}") from error

    if rules.min_shadow_requests < 1:
        raise PromotionRulesError("min_shadow_requests must be at least 1")
    if rules.min_reason_length < 1:
        raise PromotionRulesError("min_reason_length must be at least 1")
    return rules
