"""JWT authentication with role claims.

`project_docs/TECH_STACK.md` specifies PyJWT with `clinician`/`ops` role claims,
the point being governance separation: whoever *sees* a risk score is not
automatically whoever *promotes* a model. Week 5 has no ops endpoints yet, so
`require_role` exists and is tested but only `clinician` access is wired up.

Kept free of any prediction logic so both halves stay testable on their own.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.config import Settings, get_settings

CLINICIAN_ROLE = "clinician"
OPS_ROLE = "ops"
VALID_ROLES = (CLINICIAN_ROLE, OPS_ROLE)

# auto_error=False so a missing header produces our own 401 with a consistent
# body, rather than FastAPI's default 403.
bearer_scheme = HTTPBearer(auto_error=False, description="JWT bearer token")


@dataclass(frozen=True)
class Principal:
    """The authenticated caller, taken only from verified token claims."""

    subject: str
    role: str


def create_access_token(
    subject: str,
    role: str = CLINICIAN_ROLE,
    settings: Settings | None = None,
    expires_in_minutes: int | None = None,
) -> str:
    """Mints a token. Used by the dev-token script and by the tests."""
    settings = settings or get_settings()
    if role not in VALID_ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {VALID_ROLES}")

    now = datetime.now(UTC)
    minutes = settings.jwt_expiry_minutes if expires_in_minutes is None else expires_in_minutes
    payload = {
        "sub": subject,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def decode_token(token: str, settings: Settings) -> dict:
    """Verifies signature, expiry, issuer and audience.

    The algorithm is pinned to the configured one: accepting whatever the token
    header claims is how `alg: none` and HS/RS confusion attacks get in.
    """
    try:
        return jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options={"require": ["exp", "sub", "iss", "aud"]},
        )
    except jwt.ExpiredSignatureError as error:
        raise _unauthorized("Token has expired") from error
    except jwt.InvalidSignatureError as error:
        raise _unauthorized("Token signature is invalid") from error
    except jwt.InvalidTokenError as error:
        # Covers malformed tokens, bad issuer/audience, and missing claims.
        # The message stays generic on purpose -- it is returned to the caller.
        raise _unauthorized("Token is invalid") from error


def get_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> Principal:
    """FastAPI dependency resolving the caller, or raising 401."""
    if credentials is None or not credentials.credentials:
        raise _unauthorized("Authentication required")
    if credentials.scheme.lower() != "bearer":
        raise _unauthorized("Authentication required")

    claims = decode_token(credentials.credentials, settings)
    role = claims.get("role")
    if role not in VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token does not carry a recognised role",
        )
    return Principal(subject=str(claims["sub"]), role=str(role))


def require_role(*allowed: str):
    """Dependency factory gating an endpoint on role.

    Week 5 uses this for `clinician`; the `ops` endpoints it also covers arrive
    with the approval flow in Week 9.
    """

    def dependency(principal: Principal = Depends(get_principal)) -> Principal:
        if principal.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This endpoint requires a different role",
            )
        return principal

    return dependency
