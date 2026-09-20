"""The single sklearn Pipeline used identically at training and serving time.

Per ARCHITECTURE.md §3.2, this is the *only* transformation path — using the
same fitted Pipeline object at train and inference time eliminates
training/serving skew by construction.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from ml.data.clean import RAW_TARGET_COLUMN, TARGET_COLUMN
from ml.data.icd9 import map_icd9_series_to_chapter

MEDICATION_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

AGE_BANDS = [f"[{i}-{i + 10})" for i in range(0, 100, 10)]

# Admission-source risk grouping per the dataset's published IDs_mapping.csv.
ADMISSION_SOURCE_GROUPS = {
    1: "referral",
    2: "referral",
    3: "referral",
    7: "emergency",
    4: "transfer",
    5: "transfer",
    6: "transfer",
    10: "transfer",
    22: "transfer",
    25: "transfer",
    26: "transfer",
    11: "delivery_birth",
    12: "delivery_birth",
    13: "delivery_birth",
    14: "delivery_birth",
    23: "delivery_birth",
    24: "delivery_birth",
    8: "legal_other",
}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds engineered columns; does not drop or encode anything (that's the
    ColumnTransformer stage that follows it in the Pipeline)."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        df["service_utilization"] = (
            df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
        )

        med_change_mask = df[MEDICATION_COLUMNS].isin(["Up", "Down"])
        df["num_med_changes"] = med_change_mask.sum(axis=1)
        df["insulin_changed"] = df["insulin"].isin(["Up", "Down"]).astype(int)

        df["procedure_rate"] = df["num_procedures"] / df["time_in_hospital"]

        for col in ("diag_1", "diag_2", "diag_3"):
            df[f"{col}_chapter"] = map_icd9_series_to_chapter(df[col])

        df["admission_source_group"] = (
            df["admission_source_id"].map(ADMISSION_SOURCE_GROUPS).fillna("other")
        )

        return df


ENGINEERED_NUMERIC_COLUMNS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "service_utilization",
    "num_med_changes",
    "insulin_changed",
    "procedure_rate",
]

ENGINEERED_CATEGORICAL_COLUMNS = [
    "race",
    "gender",
    "age",
    "payer_code",
    "medical_specialty",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
    "diag_1_chapter",
    "diag_2_chapter",
    "diag_3_chapter",
    "admission_source_group",
    *MEDICATION_COLUMNS,
]


def build_feature_pipeline() -> Pipeline:
    """A fresh, unfitted Pipeline: engineer -> impute/encode.

    Call .fit_transform(df) at training time and .transform(df) (on the same
    fitted instance) at serving time -- never refit at serve time.
    """
    encode_stage = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median"))]),
                ENGINEERED_NUMERIC_COLUMNS,
            ),
            (
                "age_ordinal",
                OrdinalEncoder(
                    categories=[AGE_BANDS], handle_unknown="use_encoded_value", unknown_value=-1
                ),
                ["age"],
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                [c for c in ENGINEERED_CATEGORICAL_COLUMNS if c != "age"],
            ),
        ]
    )

    return Pipeline(
        [
            ("engineer", FeatureEngineer()),
            ("encode", encode_stage),
        ]
    )


# Columns that are never features. Identifiers carry no signal and would leak
# patient identity across splits; both target columns are excluded so that no
# "everything except y" selection can hand the answer back to the model.
ID_COLUMNS = ["encounter_id", "patient_nbr"]
NON_FEATURE_COLUMNS = [*ID_COLUMNS, TARGET_COLUMN, RAW_TARGET_COLUMN]


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Explicit X/y separation -- the only supported way to build a training matrix.

    Raises if the target is absent rather than silently returning an unlabelled
    frame, and drops every non-feature column whether or not it is present.
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"{TARGET_COLUMN!r} not found; was ml.data.clean.clean() applied?")

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns])
    return X, y
