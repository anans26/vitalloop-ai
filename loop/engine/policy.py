"""Loading and validating a versioned policy file.

ARCHITECTURE.md §3.8: "Every policy version is a tagged code artifact; the
Decision Card records which policy version produced it." That is what this
module enforces -- a `Policy` is immutable, carries its own version string, and
is the only place the engine gets a number from. There is no default threshold
anywhere in `loop/engine/`: a missing key in the YAML is a load-time failure,
not a silent fallback to something a reviewer never approved.

A new policy is a new file, and an old one is never edited once cards reference
it: `decision_cards.policy_version` is only meaningful if the artifact it names
still says what it said at emission. `policy-v1` is therefore retained
unchanged as ARCHITECTURE.md §3.8's literal transcription, and `policy-v2` --
the same six rules with the breach threshold calibrated on the no-drift
control, as the roadmap requires -- is what the system decides with.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ml.config import PROJECT_ROOT

POLICY_DIR = PROJECT_ROOT / "configs"

# The policy in force. policy-v1 remains on disk and loadable -- cards emitted
# under it must stay re-derivable -- but policy-v2 carries the control
# calibration the roadmap's Week 6 risk line requires, so it is what new
# decisions are taken under.
DEFAULT_POLICY_VERSION = "policy-v2"


class PolicyError(RuntimeError):
    """The policy file is missing, malformed, or incomplete. Never swallowed."""


@dataclass(frozen=True)
class ConfidenceWeights:
    """w1..w4 from §3.8, plus the two scales the terms normalise against."""

    severity: float
    breadth: float
    persistence: float
    evidence: float
    severity_scale: float
    persistence_scale: float
    matured_labels: float
    leading_indicators: float

    @property
    def total(self) -> float:
        return self.severity + self.breadth + self.persistence + self.evidence


@dataclass(frozen=True)
class Policy:
    """One loaded policy version. Immutable, and the engine's only source of numbers."""

    version: str
    description: str
    # Thresholds
    psi_breach: float
    psi_severe: float
    consecutive_windows: int
    alert_only_max_features: int
    auto_proceed_confidence: float
    matured_auroc_drop: float
    retrain_cooldown_days: int
    retrain_budget: int
    # Confidence
    weights: ConfidenceWeights
    # Rule handling
    precedence: tuple[int, ...]
    uncovered_breach_action: str
    # Recorded on every card
    acceptance_criteria: dict[str, float]

    def summary(self) -> dict[str, Any]:
        """The thresholds a card records, so a reader need not open the YAML."""
        return {
            "psi_breach": self.psi_breach,
            "psi_severe": self.psi_severe,
            "consecutive_windows": self.consecutive_windows,
            "auto_proceed_confidence": self.auto_proceed_confidence,
        }


def policy_path(version: str = DEFAULT_POLICY_VERSION, directory: Path = POLICY_DIR) -> Path:
    return directory / f"{version}.yaml"


def available_policies(directory: Path = POLICY_DIR) -> tuple[str, ...]:
    """Every policy version on disk, oldest first.

    A policy file is never removed once a card names it, so this is also the
    set of versions whose cards can still be re-derived.
    """
    return tuple(sorted(path.stem for path in directory.glob("policy-v*.yaml")))


def _require(mapping: dict, key: str, where: str):
    if key not in mapping:
        raise PolicyError(f"{where} is missing required key {key!r}")
    return mapping[key]


def parse_policy(document: dict, *, source: str = "<dict>") -> Policy:
    """Validates a loaded document into a `Policy`, or raises `PolicyError`."""
    if not isinstance(document, dict):
        raise PolicyError(f"{source} must contain a mapping at the top level")

    thresholds = _require(document, "thresholds", source)
    confidence = _require(document, "confidence", source)
    weights = _require(confidence, "weights", f"{source}:confidence")
    evidence = _require(confidence, "evidence", f"{source}:confidence")

    policy = Policy(
        version=str(_require(document, "version", source)),
        description=str(document.get("description", "")).strip(),
        psi_breach=float(_require(thresholds, "psi_breach", f"{source}:thresholds")),
        psi_severe=float(_require(thresholds, "psi_severe", f"{source}:thresholds")),
        consecutive_windows=int(
            _require(thresholds, "consecutive_windows", f"{source}:thresholds")
        ),
        alert_only_max_features=int(
            _require(thresholds, "alert_only_max_features", f"{source}:thresholds")
        ),
        auto_proceed_confidence=float(
            _require(thresholds, "auto_proceed_confidence", f"{source}:thresholds")
        ),
        matured_auroc_drop=float(
            _require(thresholds, "matured_auroc_drop", f"{source}:thresholds")
        ),
        retrain_cooldown_days=int(
            _require(thresholds, "retrain_cooldown_days", f"{source}:thresholds")
        ),
        retrain_budget=int(_require(thresholds, "retrain_budget", f"{source}:thresholds")),
        weights=ConfidenceWeights(
            severity=float(_require(weights, "severity", f"{source}:weights")),
            breadth=float(_require(weights, "breadth", f"{source}:weights")),
            persistence=float(_require(weights, "persistence", f"{source}:weights")),
            evidence=float(_require(weights, "evidence", f"{source}:weights")),
            severity_scale=float(_require(confidence, "severity_scale", f"{source}:confidence")),
            persistence_scale=float(
                _require(confidence, "persistence_scale", f"{source}:confidence")
            ),
            matured_labels=float(_require(evidence, "matured_labels", f"{source}:evidence")),
            leading_indicators=float(
                _require(evidence, "leading_indicators", f"{source}:evidence")
            ),
        ),
        precedence=tuple(int(rule) for rule in _require(document, "precedence", source)),
        uncovered_breach_action=str(_require(document, "uncovered_breach_action", source)),
        acceptance_criteria={
            str(key): float(value)
            for key, value in _require(document, "acceptance_criteria", source).items()
        },
    )
    _validate(policy, source)
    return policy


def _validate(policy: Policy, source: str) -> None:
    """Refuses a policy that could not produce a defensible decision."""
    if not policy.version:
        raise PolicyError(f"{source} has an empty version")
    if not 0.0 < policy.psi_breach < policy.psi_severe:
        raise PolicyError(
            f"{source}: psi_breach must be positive and below psi_severe "
            f"(got {policy.psi_breach} and {policy.psi_severe})"
        )
    if policy.consecutive_windows < 2:
        raise PolicyError(
            f"{source}: consecutive_windows must be at least 2 -- a persistence rule that "
            "fires on a single window is not a persistence rule"
        )
    if not 0.0 <= policy.auto_proceed_confidence <= 1.0:
        raise PolicyError(f"{source}: auto_proceed_confidence must be within [0, 1]")
    if abs(policy.weights.total - 1.0) > 1e-9:
        raise PolicyError(
            f"{source}: confidence weights must sum to 1.0 (got {policy.weights.total})"
        )
    if policy.weights.severity_scale <= 0 or policy.weights.persistence_scale <= 0:
        raise PolicyError(f"{source}: confidence scales must be positive")
    if sorted(policy.precedence) != [1, 2, 3, 4, 5]:
        raise PolicyError(
            f"{source}: precedence must order exactly rules 1-5 (rule 6 is a downgrade, "
            f"applied after the table); got {list(policy.precedence)}"
        )
    if not policy.acceptance_criteria:
        raise PolicyError(f"{source}: acceptance_criteria must not be empty")


def load_policy(version: str = DEFAULT_POLICY_VERSION, *, directory: Path = POLICY_DIR) -> Policy:
    """Reads `configs/<version>.yaml`. Raises `PolicyError` if it cannot be trusted."""
    path = policy_path(version, directory)
    if not path.exists():
        raise PolicyError(f"No policy file at {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise PolicyError(f"{path} is not valid YAML: {error}") from error

    policy = parse_policy(document, source=path.name)
    if policy.version != version:
        raise PolicyError(
            f"{path.name} declares version {policy.version!r}, which does not match its "
            "filename -- a card's policy_version must name a file a reader can open"
        )
    return policy
