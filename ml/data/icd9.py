"""Maps raw ICD-9 diagnosis codes to ~18 clinically meaningful chapter groups.

Diabetes (250.xx) is carved out of its parent chapter (240-279, endocrine/
metabolic) into its own group since it is the clinically dominant code for
this dataset's population (per DATASET_ANALYSIS.md feature-engineering notes).
"""

import pandas as pd

_RANGE_CHAPTERS = [
    (1, 139, "infectious_parasitic"),
    (140, 239, "neoplasms"),
    (240, 249, "endocrine_metabolic_other"),
    (251, 279, "endocrine_metabolic_other"),
    (280, 289, "blood"),
    (290, 319, "mental"),
    (320, 389, "nervous_sense_organs"),
    (390, 459, "circulatory"),
    (460, 519, "respiratory"),
    (520, 579, "digestive"),
    (580, 629, "genitourinary"),
    (630, 679, "pregnancy_childbirth"),
    (680, 709, "skin"),
    (710, 739, "musculoskeletal"),
    (740, 759, "congenital_anomalies"),
    (760, 779, "perinatal"),
    (780, 799, "symptoms_ill_defined"),
    (800, 999, "injury_poisoning"),
]


def _map_one(code) -> str:
    if pd.isna(code):
        return "missing"
    code = str(code).strip()
    if not code:
        return "missing"
    if code.upper().startswith("V"):
        return "supplementary_v"
    if code.upper().startswith("E"):
        return "external_causes_e"

    try:
        numeric = float(code)
    except ValueError:
        return "other_unrecognized"

    if 250.0 <= numeric < 251.0:
        return "diabetes"

    for low, high, chapter in _RANGE_CHAPTERS:
        if low <= numeric <= high:
            return chapter
    return "other_unrecognized"


def map_icd9_series_to_chapter(series: pd.Series) -> pd.Series:
    return series.map(_map_one)
