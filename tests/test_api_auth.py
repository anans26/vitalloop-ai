"""JWT authentication: token handling and endpoint enforcement."""

from datetime import UTC, datetime, timedelta

import jwt
import pytest

from api.auth import CLINICIAN_ROLE, OPS_ROLE, create_access_token, decode_token


def _post(client, payload, headers=None):
    return client.post("/predict", json=payload, headers=headers or {})


# ---------------------------------------------------------------------------
# Token handling
# ---------------------------------------------------------------------------
def test_created_token_round_trips_with_expected_claims(api_env):
    token = create_access_token("dr-synthetic", role=CLINICIAN_ROLE, settings=api_env)
    claims = decode_token(token, api_env)

    assert claims["sub"] == "dr-synthetic"
    assert claims["role"] == CLINICIAN_ROLE
    assert claims["iss"] == api_env.jwt_issuer
    assert claims["aud"] == api_env.jwt_audience
    assert "exp" in claims


def test_unknown_role_is_refused_at_creation(api_env):
    with pytest.raises(ValueError, match="unknown role"):
        create_access_token("someone", role="administrator", settings=api_env)


def test_token_signed_with_another_key_is_rejected(api_env):
    forged = jwt.encode(
        {
            "sub": "attacker",
            "role": CLINICIAN_ROLE,
            "iss": api_env.jwt_issuer,
            "aud": api_env.jwt_audience,
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        "a-different-secret",
        algorithm="HS256",
    )
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        decode_token(forged, api_env)
    assert error.value.status_code == 401


def test_algorithm_is_pinned_so_alg_none_is_rejected(api_env):
    """Trusting the token's own header is how `alg: none` attacks succeed."""
    unsigned = jwt.encode(
        {
            "sub": "attacker",
            "role": CLINICIAN_ROLE,
            "iss": api_env.jwt_issuer,
            "aud": api_env.jwt_audience,
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        key="",
        algorithm="none",
    )
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        decode_token(unsigned, api_env)


# ---------------------------------------------------------------------------
# Endpoint enforcement
# ---------------------------------------------------------------------------
def test_predict_requires_a_token(api_client, valid_payload):
    response = _post(api_client, valid_payload)
    assert response.status_code == 401
    assert "detail" in response.json()


def test_predict_rejects_a_malformed_token(api_client, valid_payload):
    response = _post(api_client, valid_payload, {"Authorization": "Bearer not.a.jwt"})
    assert response.status_code == 401


def test_predict_rejects_a_non_bearer_scheme(api_client, valid_payload):
    response = _post(api_client, valid_payload, {"Authorization": "Basic dXNlcjpwYXNz"})
    assert response.status_code == 401


def test_predict_rejects_an_expired_token(api_client, api_env, valid_payload):
    expired = create_access_token(
        "dr-synthetic", role=CLINICIAN_ROLE, settings=api_env, expires_in_minutes=-1
    )
    response = _post(api_client, valid_payload, {"Authorization": f"Bearer {expired}"})

    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()


def test_predict_rejects_an_invalid_signature(api_client, api_env, valid_payload):
    forged = jwt.encode(
        {
            "sub": "attacker",
            "role": CLINICIAN_ROLE,
            "iss": api_env.jwt_issuer,
            "aud": api_env.jwt_audience,
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        "wrong-secret",
        algorithm="HS256",
    )
    response = _post(api_client, valid_payload, {"Authorization": f"Bearer {forged}"})
    assert response.status_code == 401


def test_predict_rejects_a_token_whose_role_is_not_recognised(api_client, api_env, valid_payload):
    token = jwt.encode(
        {
            "sub": "someone",
            "role": "administrator",
            "iss": api_env.jwt_issuer,
            "aud": api_env.jwt_audience,
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        api_env.jwt_secret,
        algorithm=api_env.jwt_algorithm,
    )
    response = _post(api_client, valid_payload, {"Authorization": f"Bearer {token}"})
    assert response.status_code == 403


def test_predict_accepts_a_valid_clinician_token(api_client, valid_payload, auth_headers):
    assert _post(api_client, valid_payload, auth_headers).status_code == 200


def test_predict_also_accepts_the_ops_role(api_client, api_env, valid_payload):
    token = create_access_token("ops-user", role=OPS_ROLE, settings=api_env)
    response = _post(api_client, valid_payload, {"Authorization": f"Bearer {token}"})
    assert response.status_code == 200


def test_rejection_responses_never_echo_the_token(api_client, valid_payload):
    token = "Bearer aaaa.bbbb.cccc"
    response = _post(api_client, valid_payload, {"Authorization": token})
    assert "aaaa.bbbb.cccc" not in response.text
