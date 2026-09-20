import pandas as pd
import pytest

from ml.data.features import (
    ENGINEERED_CATEGORICAL_COLUMNS,
    ENGINEERED_NUMERIC_COLUMNS,
    MEDICATION_COLUMNS,
    NON_FEATURE_COLUMNS,
    build_feature_pipeline,
    split_features_target,
)
from ml.data.icd9 import map_icd9_series_to_chapter


def _sample_df(n: int = 20) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "time_in_hospital": [3] * n,
            "num_lab_procedures": [40] * n,
            "num_procedures": [1] * n,
            "num_medications": [10] * n,
            "number_outpatient": [0] * n,
            "number_emergency": [1] * n,
            "number_inpatient": [0] * n,
            "number_diagnoses": [5] * n,
            "race": ["Caucasian"] * n,
            "gender": ["Female"] * n,
            "age": ["[50-60)"] * n,
            "payer_code": ["missing"] * n,
            "medical_specialty": ["missing"] * n,
            "max_glu_serum": [None] * n,
            "A1Cresult": [None] * n,
            "change": ["No"] * n,
            "diabetesMed": ["Yes"] * n,
            "diag_1": ["250.83"] * n,
            "diag_2": ["V27"] * n,
            "diag_3": [None] * n,
            "admission_source_id": [7] * n,
        }
    )
    for col in MEDICATION_COLUMNS:
        df[col] = "No"
    df.loc[0, "insulin"] = "Up"
    return df


def test_icd9_chapter_mapping_carves_out_diabetes_and_handles_v_codes():
    mapped = map_icd9_series_to_chapter(pd.Series(["250.83", "V27", "486", None]))
    assert mapped.tolist() == ["diabetes", "supplementary_v", "respiratory", "missing"]


def test_pipeline_fit_transform_and_transform_produce_same_feature_count():
    """The train/serve skew guard: same fitted pipeline, same output width."""
    df = _sample_df(40)
    train, serve = df.iloc[:30], df.iloc[30:]

    pipe = build_feature_pipeline()
    X_train = pipe.fit_transform(train)
    X_serve = pipe.transform(serve)

    assert X_train.shape[1] == X_serve.shape[1]
    assert X_train.shape[0] == 30
    assert X_serve.shape[0] == 10


def test_insulin_changed_flag_detects_up_down():
    from ml.data.features import FeatureEngineer

    df = _sample_df(2)
    out = FeatureEngineer().fit_transform(df)
    assert out.loc[0, "insulin_changed"] == 1
    assert out.loc[1, "insulin_changed"] == 0


def test_split_features_target_excludes_ids_and_both_target_columns():
    df = _sample_df(5)
    df["encounter_id"] = range(5)
    df["patient_nbr"] = range(100, 105)
    df["readmitted_30d"] = [0, 1, 0, 1, 0]
    df["readmitted"] = ["NO", "<30", "NO", "<30", ">30"]

    X, y = split_features_target(df)

    assert y.tolist() == [0, 1, 0, 1, 0]
    for col in NON_FEATURE_COLUMNS:
        assert col not in X.columns


def test_split_features_target_raises_when_target_missing():
    with pytest.raises(KeyError, match="readmitted_30d"):
        split_features_target(_sample_df(3))


def test_configured_pipeline_columns_never_include_a_target_column():
    """Guards the ColumnTransformer itself, not just the caller."""
    configured = set(ENGINEERED_NUMERIC_COLUMNS) | set(ENGINEERED_CATEGORICAL_COLUMNS)
    assert configured.isdisjoint(set(NON_FEATURE_COLUMNS))
