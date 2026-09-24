"""Sends encounters from the serving stream through `/predict`, as a clinician would.

    python -m scripts.send_traffic --count 60
    python -m scripts.send_traffic --count 60 --url http://localhost:8000 --offset 500

§3.13's shadow window is "K requests" in the demo, and those requests have to
come from somewhere. This replays rows of `datasets/processed/future_stream.csv`
-- the held-out serving stream Week 2 set aside -- as real authenticated calls,
so every one produces an audit row and, while a shadow is loaded, a shadow row.

Only the columns the API's contract accepts are sent (`PredictionRequest`), so
identifiers and the label never leave this process. The token is minted with the
same secret the API verifies with, exactly as `scripts.issue_dev_token` does.
"""

import argparse
import math
import sys

import httpx
import pandas as pd

from api.auth import create_access_token
from api.config import get_settings
from api.schemas import PredictionRequest
from ml.config import PROCESSED_DIR

STREAM_PATH = PROCESSED_DIR / "future_stream.csv"

# ICD-9 codes are strings ("V27", "250.83"); read as-is, pandas turns the numeric
# ones into floats ("428" -> 428.0), which the API rightly rejects.
TEXT_COLUMNS = {"diag_1": str, "diag_2": str, "diag_3": str}


def contract_fields() -> set[str]:
    return {field.alias or name for name, field in PredictionRequest.model_fields.items()}


def payloads(frame: pd.DataFrame) -> list[dict]:
    """Rows reduced to the API's contract, with NaN sent as null."""
    fields = contract_fields()
    rows = []
    for record in frame[[c for c in frame.columns if c in fields]].to_dict(orient="records"):
        rows.append(
            {
                key: (None if isinstance(value, float) and math.isnan(value) else value)
                for key, value in record.items()
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay serving-stream rows through /predict.")
    parser.add_argument("--count", type=int, default=60, help="requests to send")
    parser.add_argument("--offset", type=int, default=0, help="first stream row to send")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--subject", default="traffic-replay", help="token subject")
    args = parser.parse_args(argv)

    frame = pd.read_csv(
        STREAM_PATH, skiprows=range(1, args.offset + 1), nrows=args.count, dtype=TEXT_COLUMNS
    )
    token = create_access_token(args.subject, role="clinician", settings=get_settings())
    headers = {"Authorization": f"Bearer {token}"}

    ok = failed = 0
    versions = set()
    with httpx.Client(base_url=args.url, headers=headers, timeout=30.0) as client:
        for body in payloads(frame):
            response = client.post("/predict", json=body)
            if response.status_code == 200:
                ok += 1
                versions.add(response.json()["model_version"])
            else:
                failed += 1

    print(f"{ok} scored, {failed} failed; served by model version(s) {sorted(versions)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
