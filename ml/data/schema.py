"""pandera schema for the raw UCI Diabetes 130-US extract.

Value domains below were read directly off the downloaded dataset (see
notebooks/01_eda.ipynb), not assumed from the docs.
"""

from pandera.pandas import Check, Column, DataFrameSchema

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

MEDICATION_VALUES = {"No", "Steady", "Up", "Down"}

_medication_fields = {
    col: Column(str, Check.isin(MEDICATION_VALUES), nullable=True) for col in MEDICATION_COLUMNS
}

raw_diabetes_schema = DataFrameSchema(
    {
        "encounter_id": Column(int, unique=True),
        "patient_nbr": Column(int),
        "race": Column(str, nullable=True),
        "gender": Column(str, Check.isin({"Female", "Male", "Unknown/Invalid"})),
        "age": Column(str, Check.str_matches(r"^\[\d+-\d+\)$")),
        "weight": Column(str, nullable=True),
        "admission_type_id": Column(int, Check.in_range(1, 8)),
        "discharge_disposition_id": Column(int, Check.in_range(1, 30)),
        "admission_source_id": Column(int, Check.in_range(1, 30)),
        "time_in_hospital": Column(int, Check.in_range(1, 14)),
        "payer_code": Column(str, nullable=True),
        "medical_specialty": Column(str, nullable=True),
        "num_lab_procedures": Column(int, Check.ge(0)),
        "num_procedures": Column(int, Check.ge(0)),
        "num_medications": Column(int, Check.ge(0)),
        "number_outpatient": Column(int, Check.ge(0)),
        "number_emergency": Column(int, Check.ge(0)),
        "number_inpatient": Column(int, Check.ge(0)),
        "diag_1": Column(str, nullable=True),
        "diag_2": Column(str, nullable=True),
        "diag_3": Column(str, nullable=True),
        "number_diagnoses": Column(int, Check.ge(1)),
        "max_glu_serum": Column(str, Check.isin({">200", ">300", "Norm"}), nullable=True),
        "A1Cresult": Column(str, Check.isin({">7", ">8", "Norm"}), nullable=True),
        **_medication_fields,
        "change": Column(str, Check.isin({"Ch", "No"})),
        "diabetesMed": Column(str, Check.isin({"Yes", "No"})),
        "readmitted": Column(str, Check.isin({"NO", ">30", "<30"})),
    },
    coerce=True,
)


def validate_raw(df):
    """Returns the validated dataframe, or raises pandera.errors.SchemaError."""
    return raw_diabetes_schema.validate(df, lazy=True)
