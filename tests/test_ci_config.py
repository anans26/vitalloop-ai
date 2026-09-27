"""Week 11: the CI pipeline and the Compose profiles, as configuration contracts.

The pipeline itself only runs on GitHub; these tests hold its *shape* to
ARCHITECTURE.md §5 -- "lint (ruff) -> unit tests -> training smoke run on a
data sample -> gate check -> image build" -- and keep it CI-safe: no service
credentials, no model download, no Ollama. The Compose tests hold the
"ollama is an optional profile" line and the dashboard's no-secret rule.
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CI = ROOT / ".github" / "workflows" / "ci.yml"
COMPOSE = ROOT / "docker" / "docker-compose.yml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(CI.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def steps(job: dict) -> str:
    return "\n".join(str(step.get("run", "")) for step in job["steps"])


def test_the_stages_run_in_the_documented_order(workflow):
    jobs = workflow["jobs"]
    assert list(jobs) == ["lint", "test", "smoke", "images"]
    assert "needs" not in jobs["lint"]
    assert jobs["test"]["needs"] == "lint"
    assert jobs["smoke"]["needs"] == "test"
    assert jobs["images"]["needs"] == "smoke"


def test_lint_runs_ruff_check_and_format(workflow):
    run = steps(workflow["jobs"]["lint"])
    assert "ruff check ." in run
    assert "ruff format --check ." in run


def test_tests_hold_each_package_to_80_percent(workflow):
    run = steps(workflow["jobs"]["test"])
    assert "pytest" in run and "--cov-branch" in run
    for package in ("loop/engine/*", "loop/gate/*", "api/*"):
        assert f"coverage report --include='{package}' --fail-under=80" in run


def test_the_smoke_run_trains_on_the_roadmaps_sample_and_gates(workflow):
    run = steps(workflow["jobs"]["smoke"])
    assert "python -m scripts.ci_smoke --rows 5000" in run


def test_images_are_built_and_both_profiles_validated(workflow):
    run = steps(workflow["jobs"]["images"])
    assert "cp docker/.env.example docker/.env" in run
    assert "docker compose config --quiet" in run
    assert "docker compose --profile ollama config --quiet" in run
    assert "docker compose build api monitor dashboard" in run


def test_ci_needs_no_service_no_secret_and_no_model(workflow):
    text = CI.read_text(encoding="utf-8")
    for job in workflow["jobs"].values():
        assert "services" not in job
    for forbidden in ("ollama pull", "secrets.", "POSTGRES_PASSWORD", "JWT_SECRET", "dvc pull"):
        assert forbidden not in text
    # Every dependency install uses the pinned constraints.
    assert text.count("pip install -r requirements.txt -c constraints.txt") == 3


def test_ollama_is_an_optional_profile_nothing_depends_on(compose):
    services = compose["services"]
    assert services["ollama"]["profiles"] == ["ollama"]
    for name, service in services.items():
        assert "ollama" not in (service.get("depends_on") or {}), name
        assert "profiles" not in service or name == "ollama"


def test_template_narration_is_the_compose_default(compose):
    for name in ("monitor", "dashboard"):
        env = compose["services"][name]["environment"]
        assert env["VITALLOOP_NARRATION_BACKEND"] == "${VITALLOOP_NARRATION_BACKEND:-template}"


def test_only_the_api_holds_the_jwt_secret(compose):
    for name, service in compose["services"].items():
        env = service.get("environment") or {}
        holds = any("JWT_SECRET" in key for key in env)
        assert holds == (name == "api"), name


def test_the_mlflow_image_is_pinned(compose):
    image = compose["services"]["mlflow"]["image"]
    assert not image.endswith(":latest") and ":" in image


def test_committed_env_template_has_placeholders_only():
    template = (ROOT / "docker" / ".env.example").read_text(encoding="utf-8")
    values = dict(
        line.split("=", 1)
        for line in template.splitlines()
        if line and not line.startswith("#") and "=" in line
    )
    assert values["VITALLOOP_JWT_SECRET"] == "generate-me"
    assert values["POSTGRES_PASSWORD"] == "changeme"
    assert values["VITALLOOP_NARRATION_BACKEND"] == "template"


def test_demo_state_archives_stay_out_of_git_and_images():
    assert "backups/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "backups" in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()


def test_no_test_can_write_the_real_alias_audit_trail(tmp_path):
    import ml.registry
    from ml.tracking_config import REGISTRY_AUDIT_PATH

    # tests/conftest.py's autouse fixture: every test gets a private trail.
    assert ml.registry.REGISTRY_AUDIT_PATH != REGISTRY_AUDIT_PATH
    assert ml.registry.REGISTRY_AUDIT_PATH.parent == tmp_path
