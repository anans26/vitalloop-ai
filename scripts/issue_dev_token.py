"""Mints a short-lived development JWT.

    python -m scripts.issue_dev_token --subject dr.smith --role clinician

Reads VITALLOOP_JWT_SECRET from the environment, so it signs with the same key
the API verifies with. Tokens are printed to stdout and never written to disk.

Week 11: when the variable is not set, it is read from `docker/.env` -- the
file Compose gives the API -- so the README quickstart needs no export step.
A value already in the environment always wins.
"""

import argparse

from api.auth import VALID_ROLES, create_access_token
from api.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue a development JWT.")
    parser.add_argument("--subject", default="dev-user", help="Token subject (caller identity).")
    parser.add_argument("--role", default="clinician", choices=VALID_ROLES)
    parser.add_argument("--minutes", type=int, default=None, help="Override expiry in minutes.")
    args = parser.parse_args()

    from scripts.seed_demo import load_env_file

    load_env_file()
    settings = get_settings()
    token = create_access_token(
        subject=args.subject,
        role=args.role,
        settings=settings,
        expires_in_minutes=args.minutes,
    )
    print(token)


if __name__ == "__main__":
    main()
