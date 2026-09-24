"""The dashboard's only write path: the API's `/ops` endpoints, under the user's token.

Every *decision* the dashboard offers -- authorise a retrain, approve or reject
a promotion -- goes through the API rather than straight into the database.
Three Week 9 properties depend on that and none of them survive a shortcut:

* **Identity comes from a verified token.** §3.13's approval row records *who*;
  the API takes that from the token's subject, so the dashboard never names the
  approver and never holds the signing secret that could mint one.
* **The `ops` role is enforced** where it is defined, in `api.auth`.
* **The champion changes live.** An approved promotion reloads the serving
  model inside the API process; a database write from here could not.

The dashboard therefore holds a pasted token, forwards it, and reports what
the API said -- including its refusals, verbatim.
"""

import httpx


class ApiError(RuntimeError):
    """The API refused or failed. `status` and `detail` are what it said."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class OpsApi:
    def __init__(self, base_url: str, token: str | None, *, timeout: float = 120.0, client=None):
        self.base_url = base_url.rstrip("/")
        self.token = (token or "").strip()
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)

    # -- plumbing -------------------------------------------------------------
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _call(self, method: str, path: str, **kwargs):
        try:
            response = self._client.request(method, path, headers=self._headers(), **kwargs)
        except httpx.HTTPError as error:
            raise ApiError(
                0, f"API unreachable at {self.base_url} ({type(error).__name__})"
            ) from error
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(response.status_code, str(detail))
        return response.json()

    # -- identity and serving ---------------------------------------------------
    def whoami(self) -> dict:
        return self._call("GET", "/ops/whoami")

    def ready(self) -> dict:
        try:
            return self._client.get("/ready").json()
        except (httpx.HTTPError, ValueError) as error:
            raise ApiError(
                0, f"API unreachable at {self.base_url} ({type(error).__name__})"
            ) from error

    def shadow(self) -> dict:
        return self._call("GET", "/ops/shadow")

    def reload(self) -> dict:
        return self._call("POST", "/ops/models/reload")

    def predict(self, payload: dict) -> dict:
        return self._call("POST", "/predict", json=payload)

    # -- decisions --------------------------------------------------------------
    def pending_retrains(self) -> list[dict]:
        return self._call("GET", "/ops/retrains/pending")

    def decide_retrain(self, card_id: str, decision: str, reason: str) -> dict:
        return self._call(
            "POST",
            f"/ops/retrains/{card_id}/decision",
            json={"decision": decision, "reason": reason},
        )

    def pending_promotions(self) -> list[dict]:
        return self._call("GET", "/ops/promotions/pending")

    def decide_promotion(self, run_id: str, decision: str, reason: str) -> dict:
        return self._call(
            "POST",
            f"/ops/promotions/{run_id}/decision",
            json={"decision": decision, "reason": reason},
        )
