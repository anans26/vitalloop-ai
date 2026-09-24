"""The alias operations a human decision may perform, behind one small interface.

The promotion code never talks to MLflow directly. It is handed an
`AliasRegistry`, which the API builds from its settings and the tests replace
with an in-memory fake -- so "approve moves champion, reject moves nothing but
shadow" is asserted without a tracking server, the same way Week 8's runner
takes its alias setter as an argument.

The MLflow implementation delegates to `ml.registry`, which remains the only
place an alias moves and the only writer of the alias audit trail.
"""

from typing import Protocol

from mlflow.tracking import MlflowClient

from ml import registry


class AliasRegistry(Protocol):
    def version_of(self, alias: str) -> str | None: ...

    def set(self, alias: str, version: str, *, run_id: str, reason: str, actor: str) -> dict: ...

    def clear(self, alias: str, *, run_id: str, reason: str, actor: str) -> dict | None: ...


class MlflowAliasRegistry:
    """`AliasRegistry` over the MLflow Model Registry, audited by `ml.registry`."""

    def __init__(self, tracking_uri: str, model_name: str, audit_path=None):
        self._client = MlflowClient(tracking_uri)
        self._model_name = model_name
        self._audit_path = audit_path

    def version_of(self, alias: str) -> str | None:
        return registry.current_alias_version(self._client, self._model_name, alias)

    def set(self, alias: str, version: str, *, run_id: str, reason: str, actor: str) -> dict:
        return registry.set_alias(
            self._client,
            model_name=self._model_name,
            alias=alias,
            version=version,
            run_id=run_id,
            reason=reason,
            audit_path=self._audit_path,
            actor=actor,
        )

    def clear(self, alias: str, *, run_id: str, reason: str, actor: str) -> dict | None:
        return registry.delete_alias(
            self._client,
            model_name=self._model_name,
            alias=alias,
            run_id=run_id,
            reason=reason,
            audit_path=self._audit_path,
            actor=actor,
        )
