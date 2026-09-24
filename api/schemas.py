"""Request and response contracts for the prediction endpoint.

The request mirrors the model's *actual* inference contract: the 44 raw columns
`ml.data.features.build_feature_pipeline` consumes, and nothing else. Two
columns present in the processed data are deliberately absent --
`admission_type_id` and `discharge_disposition_id` are used during cleaning
(the latter removes expired/hospice encounters) but never reach the model, so
asking a caller for them would misrepresent the contract.

Domain constraints are taken from the pandera schema in `ml/data/schema.py`, so
the API rejects at the boundary what the training pipeline would have rejected.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MedicationStatus = Literal["No", "Steady", "Up", "Down"]
Gender = Literal["Female", "Male", "Unknown/Invalid"]
GlucoseResult = Literal[">200", ">300", "Norm"]
A1CResult = Literal[">7", ">8", "Norm"]


class PredictionRequest(BaseModel):
    """One encounter to score.

    Contains no patient identifier by design: `encounter_id` and `patient_nbr`
    are excluded from the feature contract (Week 2), and the audit trail records
    a hash of this payload rather than the payload itself.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    # --- utilisation and stay ------------------------------------------------
    time_in_hospital: int = Field(ge=1, le=14, description="Days in hospital, 1-14.")
    num_lab_procedures: int = Field(ge=0, le=200, description="Lab procedures performed.")
    num_procedures: int = Field(ge=0, le=20, description="Non-lab procedures performed.")
    num_medications: int = Field(ge=0, le=200, description="Distinct medications administered.")
    number_outpatient: int = Field(ge=0, le=100, description="Prior outpatient visits.")
    number_emergency: int = Field(ge=0, le=100, description="Prior emergency visits.")
    number_inpatient: int = Field(ge=0, le=100, description="Prior inpatient admissions.")
    number_diagnoses: int = Field(ge=1, le=20, description="Diagnoses recorded.")
    admission_source_id: int = Field(ge=1, le=30, description="Admission source code, 1-30.")

    # --- demographics and administration -------------------------------------
    race: str | None = Field(default=None, max_length=64)
    gender: Gender
    age: str = Field(pattern=r"^\[\d+-\d+\)$", description='Ten-year band, e.g. "[60-70)".')
    payer_code: str | None = Field(default=None, max_length=32)
    medical_specialty: str | None = Field(default=None, max_length=128)

    # --- labs and medication policy ------------------------------------------
    max_glu_serum: GlucoseResult | None = Field(default=None)
    A1Cresult: A1CResult | None = Field(default=None)
    change: Literal["Ch", "No"] = Field(description='Whether medication changed ("Ch" or "No").')
    diabetesMed: Literal["Yes", "No"] = Field(description="Whether any diabetic medication given.")

    # --- diagnoses (ICD-9 codes, mapped to chapters by the pipeline) ----------
    diag_1: str | None = Field(default=None, max_length=16)
    diag_2: str | None = Field(default=None, max_length=16)
    diag_3: str | None = Field(default=None, max_length=16)

    # --- per-drug prescription status ----------------------------------------
    # All take one of: No / Steady / Up / Down (see MedicationStatus).
    metformin: MedicationStatus = "No"
    repaglinide: MedicationStatus = "No"
    nateglinide: MedicationStatus = "No"
    chlorpropamide: MedicationStatus = "No"
    glimepiride: MedicationStatus = "No"
    acetohexamide: MedicationStatus = "No"
    glipizide: MedicationStatus = "No"
    glyburide: MedicationStatus = "No"
    tolbutamide: MedicationStatus = "No"
    pioglitazone: MedicationStatus = "No"
    rosiglitazone: MedicationStatus = "No"
    acarbose: MedicationStatus = "No"
    miglitol: MedicationStatus = "No"
    troglitazone: MedicationStatus = "No"
    tolazamide: MedicationStatus = "No"
    examide: MedicationStatus = "No"
    citoglipton: MedicationStatus = "No"
    insulin: MedicationStatus = "No"
    glyburide_metformin: MedicationStatus = Field("No", alias="glyburide-metformin")
    glipizide_metformin: MedicationStatus = Field("No", alias="glipizide-metformin")
    glimepiride_pioglitazone: MedicationStatus = Field("No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: MedicationStatus = Field("No", alias="metformin-rosiglitazone")
    metformin_pioglitazone: MedicationStatus = Field("No", alias="metformin-pioglitazone")


class ShapContribution(BaseModel):
    """One factor behind a score.

    Carries the transformed feature name and its contribution -- never the
    encounter's value for that feature, which would put clinical detail into
    responses and audit rows.
    """

    feature: str = Field(description="Transformed feature name as the model sees it.")
    contribution: float = Field(description="SHAP value; positive raises predicted risk.")


class PredictionResponse(BaseModel):
    """The scored result."""

    request_id: str = Field(description="Audit identifier; matches the persisted audit row.")
    readmission_probability: float = Field(
        ge=0.0, le=1.0, description="Calibrated probability of readmission within 30 days."
    )
    predicted_class: int = Field(ge=0, le=1, description="1 when probability >= threshold.")
    decision_threshold: float = Field(description="Threshold used for predicted_class.")
    model_name: str = Field(description="Registered model name, or the local artifact.")
    model_version: str = Field(description="Served model version.")
    data_version: str | None = Field(default=None, description="DVC hash of the training data.")
    top_factors: list[ShapContribution] = Field(
        description="Top SHAP contributors for this prediction."
    )


class HealthResponse(BaseModel):
    """Liveness. Intentionally free of configuration detail."""

    status: Literal["ok"]
    service: str


class ReadinessResponse(BaseModel):
    """Readiness of the dependencies serving actually requires."""

    status: Literal["ready", "not_ready"]
    model_loaded: bool
    database_available: bool
    model_version: str | None = None


class ErrorResponse(BaseModel):
    """Every error the API returns, in one shape."""

    detail: str = Field(description="Human-readable message. Never a traceback.")
    request_id: str | None = Field(default=None, description="Present when a request was audited.")


# ---------------------------------------------------------------------------
# Week 9 -- ops decisions
# ---------------------------------------------------------------------------
class DecisionRequest(BaseModel):
    """A person's decision. Who made it comes from the token, never from here."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["APPROVE", "REJECT"] = Field(description="APPROVE or REJECT.")
    reason: str = Field(
        min_length=1,
        max_length=1000,
        description="Why, in words. Required for both outcomes; recorded verbatim.",
    )


class ApprovalResponse(BaseModel):
    """The approval row as written, plus what it moved."""

    approval_id: str
    kind: Literal["RETRAIN", "PROMOTION"]
    card_id: str
    run_id: str | None = None
    decision: Literal["APPROVE", "REJECT"]
    approver: str
    reason: str
    rules_version: str | None = None
    challenger_version: str | None = None
    champion_version_before: str | None = None
    champion_version_after: str | None = None
    champion_alias_moved: bool = False
    shadow_alias_cleared: bool = False
    serving: dict | None = Field(
        default=None, description="Models this API instance serves after the decision."
    )
