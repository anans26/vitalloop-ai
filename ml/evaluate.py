"""Week 3 metric suite, computed on the frozen evaluation slice.

No single number is reported as "the" score. Discrimination (ROC-AUC, PR-AUC),
probability quality (Brier, log loss, ECE), and the operating points a clinician
would actually work (0.5 and the top risk decile) are reported together, overall
and per subgroup -- the subgroup breakdown exists because Week 8's promotion gate
blocks on subgroup regression, so the baseline has to be on record now.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

from ml.config import DECISION_THRESHOLD, ECE_BINS, RELIABILITY_BINS, TOP_DECILE_FRACTION


def expected_calibration_error(y_true, y_prob, n_bins: int = ECE_BINS) -> float:
    """Equal-width binned |accuracy - confidence|, weighted by bin population."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed bins so probabilities of exactly 1.0 land in the last bin
    bin_index = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_index == b
        if not mask.any():
            continue
        ece += (mask.sum() / len(y_prob)) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def recall_at_top_fraction(y_true, y_prob, fraction: float = TOP_DECILE_FRACTION) -> float:
    """Share of all positives captured in the highest-risk `fraction` of patients."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    positives = y_true.sum()
    if positives == 0:
        return float("nan")

    k = max(1, int(round(len(y_prob) * fraction)))
    top_k = np.argsort(-y_prob, kind="stable")[:k]
    return float(y_true[top_k].sum() / positives)


def threshold_counts(y_true, y_prob, threshold: float) -> dict:
    """Confusion-matrix counts plus the rates derived from them."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "threshold": round(float(threshold), 6),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "predicted_positive_rate": round(float(y_pred.mean()), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
    }


def top_decile_threshold(y_prob, fraction: float = TOP_DECILE_FRACTION) -> float:
    """The score cut that flags exactly the highest-risk `fraction` of patients."""
    return float(np.quantile(np.asarray(y_prob, dtype=float), 1.0 - fraction))


def score_predictions(y_true, y_prob) -> dict:
    """The full metric block for one set of probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    decile_cut = top_decile_threshold(y_prob)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 6),
        "average_precision": round(float(average_precision_score(y_true, y_prob)), 6),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 6),
        "log_loss": round(float(log_loss(y_true, y_prob, labels=[0, 1])), 6),
        "expected_calibration_error": round(expected_calibration_error(y_true, y_prob), 6),
        "recall_at_top_decile": round(recall_at_top_fraction(y_true, y_prob), 6),
        "positive_rate": round(float(y_true.mean()), 6),
        "counts_at_default_threshold": threshold_counts(y_true, y_prob, DECISION_THRESHOLD),
        "counts_at_top_decile_threshold": threshold_counts(y_true, y_prob, decile_cut),
    }


def reliability_table(y_true, y_prob, n_bins: int = RELIABILITY_BINS) -> pd.DataFrame:
    """Binned predicted-vs-observed frequency -- the calibration curve, as data."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_index == b
        rows.append(
            {
                "bin": b,
                "bin_lower": round(float(edges[b]), 6),
                "bin_upper": round(float(edges[b + 1]), 6),
                "count": int(mask.sum()),
                "mean_predicted": round(float(y_prob[mask].mean()), 6) if mask.any() else None,
                "observed_rate": round(float(y_true[mask].mean()), 6) if mask.any() else None,
            }
        )
    return pd.DataFrame(rows)


def subgroup_metrics(frame: pd.DataFrame, y_true, y_prob, columns) -> dict:
    """Per-subgroup ROC-AUC and positive rate, for the Week 8 fairness gate.

    A subgroup with a single outcome class has no defined AUROC; it is reported
    with a null rather than dropped, so a shrinking subgroup stays visible.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    out: dict = {}
    for column in columns:
        if column not in frame.columns:
            continue
        per_value = {}
        for value, idx in frame.groupby(column, dropna=False).indices.items():
            group_true, group_prob = y_true[idx], y_prob[idx]
            has_both_classes = len(np.unique(group_true)) == 2
            per_value[str(value)] = {
                "count": int(len(idx)),
                "positive_rate": round(float(group_true.mean()), 6),
                "roc_auc": round(float(roc_auc_score(group_true, group_prob)), 6)
                if has_both_classes
                else None,
            }
        out[column] = per_value
    return out


def _write_calibration_figure(y_true, raw_prob, calibrated_prob, path) -> None:
    """Reliability curve for raw vs calibrated probabilities.

    PNG metadata is pinned so the figure is byte-reproducible; it is a DVC stage
    output and matplotlib otherwise stamps its own version into the file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(6, 6))
    axes.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="perfectly calibrated")

    for label, probabilities in (("raw", raw_prob), ("calibrated", calibrated_prob)):
        table = reliability_table(y_true, probabilities).dropna(subset=["mean_predicted"])
        axes.plot(table["mean_predicted"], table["observed_rate"], marker="o", label=label)

    axes.set_xlabel("mean predicted probability")
    axes.set_ylabel("observed readmission rate")
    axes.set_title("Calibration on the frozen evaluation slice")
    axes.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=120, metadata={"Software": None})
    plt.close(figure)


def main() -> None:
    import json

    from ml.config import (
        CALIBRATION_FIGURE_PATH,
        EVAL_PATH,
        METRICS_PATH,
        RELIABILITY_PATH,
        REPORTS_DIR,
        SUBGROUP_COLUMNS,
    )
    from ml.data.clean import TARGET_COLUMN
    from ml.data.features import split_features_target
    from ml.train import load_models, load_split

    eval_df = load_split(EVAL_PATH)
    X_eval, y_eval = split_features_target(eval_df)
    base_model, calibrated_model = load_models()

    raw_prob = base_model.predict_proba(X_eval)[:, 1]
    calibrated_prob = calibrated_model.predict_proba(X_eval)[:, 1]

    raw_metrics = score_predictions(y_eval, raw_prob)
    calibrated_metrics = score_predictions(y_eval, calibrated_prob)

    report = {
        "evaluation_slice": "datasets/processed/eval_frozen.csv",
        "eval_rows": int(len(eval_df)),
        "target": TARGET_COLUMN,
        "positive_rate": round(float(eval_df[TARGET_COLUMN].mean()), 6),
        "inference_model": "calibrated",
        "raw": raw_metrics,
        "calibrated": calibrated_metrics,
        # Negative deltas mean calibration improved the probability estimates.
        "calibration_effect": {
            "brier_delta": round(calibrated_metrics["brier_score"] - raw_metrics["brier_score"], 6),
            "log_loss_delta": round(calibrated_metrics["log_loss"] - raw_metrics["log_loss"], 6),
            "ece_delta": round(
                calibrated_metrics["expected_calibration_error"]
                - raw_metrics["expected_calibration_error"],
                6,
            ),
            "roc_auc_delta": round(calibrated_metrics["roc_auc"] - raw_metrics["roc_auc"], 6),
        },
        "subgroups_calibrated": subgroup_metrics(
            eval_df, y_eval, calibrated_prob, SUBGROUP_COLUMNS
        ),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")

    table = reliability_table(y_eval, calibrated_prob)
    table.insert(0, "model", "calibrated")
    raw_table = reliability_table(y_eval, raw_prob)
    raw_table.insert(0, "model", "raw")
    pd.concat([raw_table, table], ignore_index=True).to_csv(
        RELIABILITY_PATH, index=False, lineterminator="\n"
    )

    _write_calibration_figure(y_eval, raw_prob, calibrated_prob, CALIBRATION_FIGURE_PATH)

    print(f"eval rows: {len(eval_df)}")
    print(
        f"  raw        ROC-AUC {raw_metrics['roc_auc']:.4f}  Brier {raw_metrics['brier_score']:.4f}"
    )
    print(
        f"  calibrated ROC-AUC {calibrated_metrics['roc_auc']:.4f}  "
        f"Brier {calibrated_metrics['brier_score']:.4f}"
    )
    print(f"metrics written to {METRICS_PATH}")


if __name__ == "__main__":
    main()
