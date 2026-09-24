"""`configs/promotion-v*.yaml`: loaded strictly, like the policy and the gate criteria."""

import pytest

from loop.approval.rules import DEFAULT_RULES_VERSION, PromotionRulesError, load_rules


def test_the_shipped_rules_load():
    rules = load_rules()
    assert rules.version == DEFAULT_RULES_VERSION == "promotion-v1"
    assert rules.min_shadow_requests >= 1
    assert rules.min_reason_length >= 1


def _write(tmp_path, body):
    (tmp_path / "promotion-vt.yaml").write_text(body, encoding="utf-8")
    return tmp_path


def test_a_missing_file_is_an_error(tmp_path):
    with pytest.raises(PromotionRulesError, match="no promotion rules"):
        load_rules("promotion-vt", tmp_path)


def test_an_unknown_key_is_an_error(tmp_path):
    body = (
        "version: promotion-vt\nmin_shadow_requests: 5\nmin_reason_length: 3\nautopromote: true\n"
    )
    with pytest.raises(PromotionRulesError, match="unknown"):
        load_rules("promotion-vt", _write(tmp_path, body))


def test_a_missing_key_is_an_error(tmp_path):
    with pytest.raises(PromotionRulesError, match="missing"):
        load_rules(
            "promotion-vt", _write(tmp_path, "version: promotion-vt\nmin_shadow_requests: 5\n")
        )


def test_the_declared_version_must_match_the_file(tmp_path):
    body = "version: promotion-v9\nmin_shadow_requests: 5\nmin_reason_length: 3\n"
    with pytest.raises(PromotionRulesError, match="declares version"):
        load_rules("promotion-vt", _write(tmp_path, body))


@pytest.mark.parametrize("key", ["min_shadow_requests", "min_reason_length"])
def test_a_zero_threshold_is_an_error(tmp_path, key):
    values = {"min_shadow_requests": 5, "min_reason_length": 3, key: 0}
    body = "version: promotion-vt\n" + "".join(f"{k}: {v}\n" for k, v in values.items())
    with pytest.raises(PromotionRulesError, match="at least 1"):
        load_rules("promotion-vt", _write(tmp_path, body))
