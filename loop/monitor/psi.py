"""Hand-rolled PSI and KS -- the cross-check, not the engine of record.

`project_docs/TECH_STACK.md` calls hand-rolled PSI "an acceptable fallback and a
good unit-test cross-check", and the Week 6 task list asks for it on two
features. This module is that implementation.

It is deliberately **not** a copy of Evidently's. Evidently bins numeric
features with Sturges edges over the pooled reference+current sample; this bins
them at reference quantiles, which is the textbook formulation. Two independent
binnings that agree on whether a feature breached is evidence about the feature.
Two identical binnings that agree is evidence about copy-paste.

Nothing in the serving or monitoring path decides anything from these numbers --
`loop/monitor/drift.py` is the engine of record. These functions exist so the
tests have something to disagree with.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Keeps log(0) and division by zero out of the sum when a bin or category is
# empty on one side. Small enough not to move a real PSI, large enough that an
# empty bin contributes a finite (and visibly large) term.
SHARE_FLOOR = 1e-6

DEFAULT_NUMERIC_BINS = 10


def psi_from_shares(reference_shares, current_shares) -> float:
    """PSI over two already-binned distributions.

    sum((current - reference) * ln(current / reference)), the standard form.
    Both inputs must be non-negative and describe the same bins in the same
    order; they are renormalised so callers can pass counts or proportions.
    """
    reference = np.asarray(reference_shares, dtype=float)
    current = np.asarray(current_shares, dtype=float)

    if reference.shape != current.shape:
        raise ValueError("reference and current must describe the same bins")
    if reference.ndim != 1 or reference.size == 0:
        raise ValueError("shares must be a non-empty one-dimensional sequence")
    if (reference < 0).any() or (current < 0).any():
        raise ValueError("shares must be non-negative")

    reference_total, current_total = reference.sum(), current.sum()
    if reference_total <= 0 or current_total <= 0:
        raise ValueError("shares must sum to a positive value on both sides")

    reference = np.clip(reference / reference_total, SHARE_FLOOR, None)
    current = np.clip(current / current_total, SHARE_FLOOR, None)

    return float(np.sum((current - reference) * np.log(current / reference)))


def numeric_bin_shares(
    reference: pd.Series, current: pd.Series, bins: int = DEFAULT_NUMERIC_BINS
) -> tuple[np.ndarray, np.ndarray]:
    """Bin both samples at the *reference* quantiles.

    The outer edges are infinite so a current value outside the reference range
    lands in the end bin instead of being silently dropped -- exactly the case
    where PSI should react.

    A reference with fewer distinct quantile edges than requested (a near-
    constant feature) collapses to the bins it actually has rather than
    producing empty ones.
    """
    reference_values = pd.Series(reference).dropna().to_numpy(dtype=float)
    current_values = pd.Series(current).dropna().to_numpy(dtype=float)
    if reference_values.size == 0 or current_values.size == 0:
        raise ValueError("both samples must contain at least one non-null value")

    edges = np.unique(np.quantile(reference_values, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        # Constant reference. Quantile bins collapse to nothing, so bin as
        # (< c), (== c), (> c) instead -- otherwise every current value would
        # land in the single surviving bin and a feature that stopped being
        # constant would score PSI 0.
        constant = edges[0]
        edges = np.array([-np.inf, constant, np.nextafter(constant, np.inf), np.inf])
    else:
        edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))

    reference_counts, _ = np.histogram(reference_values, bins=edges)
    current_counts, _ = np.histogram(current_values, bins=edges)
    return reference_counts.astype(float), current_counts.astype(float)


def numeric_psi(reference, current, bins: int = DEFAULT_NUMERIC_BINS) -> float:
    """PSI for a continuous feature, binned at reference quantiles."""
    reference_counts, current_counts = numeric_bin_shares(reference, current, bins)
    return psi_from_shares(reference_counts, current_counts)


def categorical_bin_shares(reference, current) -> tuple[np.ndarray, np.ndarray]:
    """Counts per category over the union of both samples' categories.

    Categories are sorted so the bin order -- and therefore the PSI -- does not
    depend on which values happened to appear first.
    """
    reference_counts = pd.Series(reference).dropna().astype(str).value_counts()
    current_counts = pd.Series(current).dropna().astype(str).value_counts()
    if reference_counts.empty or current_counts.empty:
        raise ValueError("both samples must contain at least one non-null value")

    categories = sorted(set(reference_counts.index) | set(current_counts.index))
    return (
        np.array([float(reference_counts.get(c, 0)) for c in categories]),
        np.array([float(current_counts.get(c, 0)) for c in categories]),
    )


def categorical_psi(reference, current) -> float:
    """PSI for a categorical feature, one bin per observed category."""
    return psi_from_shares(*categorical_bin_shares(reference, current))


def population_stability_index(
    reference, current, *, numeric: bool, bins: int | None = None
) -> float:
    """PSI for either feature kind, so callers do not branch on the type."""
    if numeric:
        return numeric_psi(reference, current, bins or DEFAULT_NUMERIC_BINS)
    return categorical_psi(reference, current)


def ks_p_value(reference, current) -> float:
    """Two-sample Kolmogorov-Smirnov p-value. Numeric features only.

    Low p-value means the two samples are unlikely to share a distribution,
    which is why `KS_P_VALUE_THRESHOLD` is an upper bound rather than a lower
    one.
    """
    reference_values = pd.Series(reference).dropna().to_numpy(dtype=float)
    current_values = pd.Series(current).dropna().to_numpy(dtype=float)
    if reference_values.size == 0 or current_values.size == 0:
        raise ValueError("both samples must contain at least one non-null value")
    return float(ks_2samp(reference_values, current_values).pvalue)
