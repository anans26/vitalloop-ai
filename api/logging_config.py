"""structlog JSON logging, per `project_docs/TECH_STACK.md`.

The stack entry reads "structlog JSON logs (no PHI -- hashes only)". That is a
constraint on what callers of this module may log, and this module makes it
easy to honour: `request_log_fields` returns the only per-request fields the
API is permitted to emit -- identifiers, the payload *hash*, and outcome
metadata. Field values from the request body never appear.
"""

import logging
import sys

import structlog


def configure_logging(level: int = logging.INFO) -> None:
    """JSON to stdout, which is what a container should emit."""
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "vitalloop.api"):
    return structlog.get_logger(name)


def request_log_fields(
    *,
    request_id: str,
    caller: str,
    input_hash: str,
    model_version: str,
    status: str,
    latency_ms: float | None = None,
    error_category: str | None = None,
) -> dict:
    """The complete set of per-request fields the API may log.

    Note what is absent: the request body, any field within it, the risk score,
    and the bearer token. The input hash is sufficient to correlate a log line
    with its audit row without duplicating clinical data into the log stream.
    """
    fields = {
        "request_id": request_id,
        "caller": caller,
        "input_hash": input_hash,
        "model_version": model_version,
        "status": status,
    }
    if latency_ms is not None:
        fields["latency_ms"] = round(latency_ms, 2)
    if error_category is not None:
        fields["error_category"] = error_category
    return fields
