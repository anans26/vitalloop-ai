"""Loading and validating a versioned gate-criteria file.

The discipline `loop/engine/policy.py` applies to the policy table applies here
for the same reason: ARCHITECTURE.md §10 puts "gate criteria" in `configs/` so
that changing a promotion threshold is a reviewable diff. There is no default
criterion anywhere in `loop/gate/` -- a missing key is a load-time failure, not
a silent fallback to a bar nobody approved.

A criteria file is never edited once a gate result references it, because
`retrain_runs.gate_result` records `criteria_version`, and that is only
meaningful if the artifact it names still says what it said at evaluation time.

**A card's own criteria outrank this file.** RESEARCH_NOVELTY.md C1 requires
the Decision Card to pin "the acceptance criteria the challenger must meet
**before** training starts", so a card-driven run is judged by the bar that was
set when the decision was taken. `from_card` builds that bar, keeping the
evaluation protocol (which sets, which subgroup columns) from the file, because
a card carries thresholds and not protocol.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ml.config import PROJECT_ROOT

CRITERIA_DIR = PROJECT_ROOT / "configs"

# The criteria version in force. gate-v1 is §3.12/§4.5's literal transcription.
DEFAULT_CRITERIA_VERSION = "gate-v1"

FROZEN_HOLDOUT = "frozen_holdout"
RECENT_LABELED_WINDOW = "recent_labeled_window"
KNOWN_SETS = (FROZEN_HOLDOUT, RECENT_LABELED_WINDOW)

# The five threshold names, which are also the keys a policy pins onto a card.
CRITERION_KEYS = (
    "auroc_non_inferiority_margin",
    "recall_top_decile_min_ratio",
    "max_brier_increase",
    "max_ece",
    "max_subgroup_auroc_drop",
)


class CriteriaError(RuntimeError):
    """The criteria file is missing, malformed, or incomplete. Never swallowed."""


@dataclass(frozen=True)
class GateCriteria:
    """One loaded criteria version. Immutable, and the gate's only source of numbers."""

    version: str
    description: str

    # §3.12 / §4.5, in the document's own order.
    auroc_non_inferiority_margin: float
    recall_top_decile_min_ratio: float
    max_brier_increase: float
    max_ece: float
    max_subgroup_auroc_drop: float

    # Protocol: what the thresholds are applied to.
    evaluation_sets: tuple[str, ...]
    require_all_sets: bool
    subgroup_columns: tuple[str, ...]
    top_decile_fraction: float

    def thresholds(self) -> dict[str, float]:
        """The five numbers, in the shape a card's `acceptance_criteria` uses."""
        return {key: float(getattr(self, key)) for key in CRITERION_KEYS}

    def summary(self) -> dict[str, Any]:
        """What a gate result records, so a reader need not open the YAML."""
        return {
            "criteria_version": self.version,
            **self.thresholds(),
            "evaluation_sets": list(self.evaluation_sets),
            "require_all_sets": self.require_all_sets,
            "subgroup_columns": list(self.subgroup_columns),
        }

    def with_thresholds(self, thresholds: dict, *, version: str) -> "GateCriteria":
        """The same protocol, judged against another artifact's numbers.

        Used for a card-driven run: the card supplies the five thresholds it
        pinned at decision time, this file supplies the evaluation protocol.
        """
        missing = [key for key in CRITERION_KEYS if key not in thresholds]
        if missing:
            raise CriteriaError(
                f"{version} is missing gate criteria {missing}; a challenger cannot be "
                "judged against a bar that was never recorded"
            )
        replaced = GateCriteria(
            version=version,
            description=self.description,
            auroc_non_inferiority_margin=float(thresholds["auroc_non_inferiority_margin"]),
            recall_top_decile_min_ratio=float(thresholds["recall_top_decile_min_ratio"]),
            max_brier_increase=float(thresholds["max_brier_increase"]),
            max_ece=float(thresholds["max_ece"]),
            max_subgroup_auroc_drop=float(thresholds["max_subgroup_auroc_drop"]),
            evaluation_sets=self.evaluation_sets,
            require_all_sets=self.require_all_sets,
            subgroup_columns=self.subgroup_columns,
            top_decile_fraction=self.top_decile_fraction,
        )
        _validate(replaced, version)
        return replaced


def criteria_path(version: str = DEFAULT_CRITERIA_VERSION, directory: Path = CRITERIA_DIR) -> Path:
    return directory / f"{version}.yaml"


def available_criteria(directory: Path = CRITERIA_DIR) -> tuple[str, ...]:
    """Every criteria version on disk, oldest first."""
    return tuple(sorted(path.stem for path in directory.glob("gate-v*.yaml")))


def _require(mapping: dict, key: str, where: str):
    if key not in mapping:
        raise CriteriaError(f"{where} is missing required key {key!r}")
    return mapping[key]


def parse_criteria(document: dict, *, source: str = "<dict>") -> GateCriteria:
    """Validates a loaded document into `GateCriteria`, or raises `CriteriaError`."""
    if not isinstance(document, dict):
        raise CriteriaError(f"{source} must contain a mapping at the top level")

    values = _require(document, "criteria", source)
    evaluation = _require(document, "evaluation", source)
    where_c = f"{source}:criteria"
    where_e = f"{source}:evaluation"

    parsed = GateCriteria(
        version=str(_require(document, "version", source)),
        description=str(document.get("description", "")).strip(),
        auroc_non_inferiority_margin=float(
            _require(values, "auroc_non_inferiority_margin", where_c)
        ),
        recall_top_decile_min_ratio=float(_require(values, "recall_top_decile_min_ratio", where_c)),
        max_brier_increase=float(_require(values, "max_brier_increase", where_c)),
        max_ece=float(_require(values, "max_ece", where_c)),
        max_subgroup_auroc_drop=float(_require(values, "max_subgroup_auroc_drop", where_c)),
        evaluation_sets=tuple(str(name) for name in _require(evaluation, "sets", where_e)),
        require_all_sets=bool(_require(evaluation, "require_all_sets", where_e)),
        subgroup_columns=tuple(
            str(column) for column in _require(evaluation, "subgroup_columns", where_e)
        ),
        top_decile_fraction=float(_require(evaluation, "top_decile_fraction", where_e)),
    )
    _validate(parsed, source)
    return parsed


def _validate(criteria: GateCriteria, source: str) -> None:
    """Refuses criteria that could not defend a promotion."""
    if not criteria.version:
        raise CriteriaError(f"{source} has an empty version")
    if criteria.auroc_non_inferiority_margin < 0:
        raise CriteriaError(
            f"{source}: auroc_non_inferiority_margin must not be negative -- a negative "
            "margin would demand a strictly better challenger, which is not §3.12's rule"
        )
    if criteria.max_brier_increase < 0:
        raise CriteriaError(f"{source}: max_brier_increase must not be negative")
    if criteria.max_subgroup_auroc_drop < 0:
        raise CriteriaError(f"{source}: max_subgroup_auroc_drop must not be negative")
    if not 0.0 < criteria.max_ece <= 1.0:
        raise CriteriaError(f"{source}: max_ece must be within (0, 1]")
    if criteria.recall_top_decile_min_ratio <= 0:
        raise CriteriaError(f"{source}: recall_top_decile_min_ratio must be positive")
    if not 0.0 < criteria.top_decile_fraction < 1.0:
        raise CriteriaError(f"{source}: top_decile_fraction must be within (0, 1)")
    if not criteria.evaluation_sets:
        raise CriteriaError(
            f"{source}: evaluation.sets must not be empty -- §3.12 gates on a frozen "
            "holdout and the most recent labeled window, never on nothing"
        )
    unknown = [name for name in criteria.evaluation_sets if name not in KNOWN_SETS]
    if unknown:
        raise CriteriaError(f"{source}: unknown evaluation set {unknown}; expected {KNOWN_SETS}")
    if not criteria.subgroup_columns:
        raise CriteriaError(
            f"{source}: evaluation.subgroup_columns must not be empty -- subgroup "
            "non-regression is a blocking criterion, not an optional report"
        )


def load_criteria(
    version: str = DEFAULT_CRITERIA_VERSION, *, directory: Path = CRITERIA_DIR
) -> GateCriteria:
    """Reads `configs/<version>.yaml`. Raises `CriteriaError` if it cannot be trusted."""
    path = criteria_path(version, directory)
    if not path.exists():
        raise CriteriaError(f"No gate criteria file at {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise CriteriaError(f"{path} is not valid YAML: {error}") from error

    criteria = parse_criteria(document, source=path.name)
    if criteria.version != version:
        raise CriteriaError(
            f"{path.name} declares version {criteria.version!r}, which does not match its "
            "filename -- a gate result's criteria_version must name a file a reader can open"
        )
    return criteria


def from_card(
    card_acceptance_criteria: dict,
    *,
    card_id: str,
    base: GateCriteria | None = None,
) -> GateCriteria:
    """The bar a specific card pinned, with the protocol from the criteria file.

    §3.9 records `acceptance_criteria` on the card at decision time, so a
    challenger is judged by that rather than by whatever `configs/` holds when
    the retrain eventually runs. The returned version is `card:<card_id>`, which
    names the artifact a reader must open to re-derive the verdict.
    """
    base = base or load_criteria()
    return base.with_thresholds(card_acceptance_criteria, version=f"card:{card_id}")
