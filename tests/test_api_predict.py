"""Prediction endpoint: response contract and request validation."""

import pytest

from api.schemas import PredictionRequest


def _predict(client, payload, headers):
    return client.post("/predict", json=payload, headers=headers)


# ---------------------------------------------------------------------------
# Response contract
# ---------------------------------------------------------------------------
def test_successful_prediction_returns_the_documented_shape(
    api_client, valid_payload, auth_headers
):
    body = _predict(api_client, valid_payload, auth_headers).json()

    assert set(body) == {
        "request_id",
        "readmission_probability",
        "predicted_class",
        "decision_threshold",
        "model_name",
        "model_version",
        "data_version",
        "top_factors",
    }


def test_probability_is_a_valid_probability(api_client, valid_payload, auth_headers):
    body = _predict(api_client, valid_payload, auth_headers).json()
    assert 0.0 <= body["readmission_probability"] <= 1.0


def test_predicted_class_follows_the_threshold(api_client, valid_payload, auth_headers):
    body = _predict(api_client, valid_payload, auth_headers).json()
    expected = int(body["readmission_probability"] >= body["decision_threshold"])

    assert body["predicted_class"] in (0, 1)
    assert body["predicted_class"] == expected


def test_response_identifies_the_served_model(api_client, valid_payload, auth_headers):
    body = _predict(api_client, valid_payload, auth_headers).json()
    assert body["model_name"] == "test-readmission"
    assert body["model_version"] == "1"
    assert body["data_version"] == "testdatahash"


def test_response_carries_top_shap_factors(api_client, valid_payload, auth_headers):
    """ARCHITECTURE.md §3.6 requires top-3 factors alongside the score."""
    body = _predict(api_client, valid_payload, auth_headers).json()
    factors = body["top_factors"]

    assert len(factors) == 3
    for factor in factors:
        assert set(factor) == {"feature", "contribution"}
        assert isinstance(factor["feature"], str)
        assert isinstance(factor["contribution"], float)


def test_explanations_never_leak_the_encounters_feature_values(
    api_client, valid_payload, auth_headers
):
    """Contributions are attributions, not the patient's values."""
    body = _predict(api_client, valid_payload, auth_headers).json()
    for factor in body["top_factors"]:
        assert "feature_value" not in factor
        assert "value" not in factor


def test_request_id_is_unique_per_call(api_client, valid_payload, auth_headers):
    first = _predict(api_client, valid_payload, auth_headers).json()["request_id"]
    second = _predict(api_client, valid_payload, auth_headers).json()["request_id"]
    assert first != second


def test_model_is_not_retrained_per_request(api_client, valid_payload, auth_headers):
    """The same fitted object must serve every request."""
    bundle_before = api_client.app.state.model_bundle
    calibrated_before = bundle_before.calibrated_model

    _predict(api_client, valid_payload, auth_headers)
    _predict(api_client, valid_payload, auth_headers)

    assert api_client.app.state.model_bundle is bundle_before
    assert api_client.app.state.model_bundle.calibrated_model is calibrated_before


def test_identical_payloads_produce_identical_scores(api_client, valid_payload, auth_headers):
    first = _predict(api_client, valid_payload, auth_headers).json()
    second = _predict(api_client, valid_payload, auth_headers).json()
    assert first["readmission_probability"] == second["readmission_probability"]


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------
def test_missing_required_field_is_rejected(api_client, valid_payload, auth_headers):
    payload = dict(valid_payload)
    del payload["time_in_hospital"]

    response = _predict(api_client, payload, auth_headers)
    assert response.status_code == 422
    assert any(e["field"] == "time_in_hospital" for e in response.json()["errors"])


def test_wrong_type_is_rejected(api_client, valid_payload, auth_headers):
    payload = dict(valid_payload, num_medications="many")
    assert _predict(api_client, payload, auth_headers).status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("time_in_hospital", 0),  # pandera range is 1-14
        ("time_in_hospital", 15),
        ("number_diagnoses", 0),  # schema requires >= 1
        ("admission_source_id", 99),  # documented range is 1-30
        ("gender", "Unspecified"),  # not in the dataset's domain
        ("age", "sixty"),  # must match the [60-70) band pattern
        ("change", "Maybe"),
        ("diabetesMed", "Sometimes"),
        ("metformin", "Increased"),  # medications are No/Steady/Up/Down
        ("A1Cresult", ">9"),
    ],
)
def test_out_of_domain_values_are_rejected(api_client, valid_payload, auth_headers, field, value):
    payload = dict(valid_payload)
    payload[field] = value
    assert _predict(api_client, payload, auth_headers).status_code == 422


def test_unknown_fields_are_rejected(api_client, valid_payload, auth_headers):
    """extra='forbid' keeps callers from smuggling identifiers into the body."""
    payload = dict(valid_payload, patient_nbr=123456)
    assert _predict(api_client, payload, auth_headers).status_code == 422


def test_validation_errors_do_not_echo_submitted_values(api_client, valid_payload, auth_headers):
    """A 422 must not put clinical values back on the wire."""
    payload = dict(valid_payload, gender="Unspecified", num_medications=987654)
    response = _predict(api_client, payload, auth_headers)

    assert response.status_code == 422
    assert "Unspecified" not in response.text
    assert "987654" not in response.text


def test_request_schema_matches_the_pipeline_feature_contract():
    """The API must ask for exactly what the model consumes -- no more, no less."""
    from ml.data.features import (
        ENGINEERED_CATEGORICAL_COLUMNS,
        ENGINEERED_NUMERIC_COLUMNS,
        MEDICATION_COLUMNS,
    )

    engineered = {
        "service_utilization",
        "num_med_changes",
        "insulin_changed",
        "procedure_rate",
        "diag_1_chapter",
        "diag_2_chapter",
        "diag_3_chapter",
        "admission_source_group",
    }
    required = (
        (set(ENGINEERED_NUMERIC_COLUMNS) | set(ENGINEERED_CATEGORICAL_COLUMNS)) - engineered
    ) | {
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "num_procedures",
        "time_in_hospital",
        "diag_1",
        "diag_2",
        "diag_3",
        "admission_source_id",
        *MEDICATION_COLUMNS,
    }
    exposed = {f.alias or name for name, f in PredictionRequest.model_fields.items()}

    assert exposed == required


def test_request_schema_excludes_identifiers_and_targets():
    exposed = {f.alias or name for name, f in PredictionRequest.model_fields.items()}
    for forbidden in ("encounter_id", "patient_nbr", "readmitted", "readmitted_30d"):
        assert forbidden not in exposed
