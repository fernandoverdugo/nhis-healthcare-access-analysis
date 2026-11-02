from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

ADULT_FILE = RAW_DIR / "adult25.csv"
INCOME_FILE = RAW_DIR / "adultinc25.csv"

BASE_COLUMNS = [
    "HHX",
    "MEDDL12M_A",
    "MEDNG12M_A",
    "AGEP_A",
    "SEX_A",
    "HISPALLP_A",
    "EDUCP_A",
    "MARSTAT_A",
    "PCNTADLT_A",
    "PCNTKIDS_A",
    "HOUTENURE_A",
    "EMPWRKLSW1_A",
    "NATUSBORN_A",
    "CITZNSTP_A",
    "NOTCOV_A",
    "USUALPL_A",
    "REGION",
    "URBRRL23",
    "PHSTAT_A",
    "ANYDIFF_A",
    "WTFA_A",
    "PSTRAT",
    "PPSU",
]

MISSING_CODES = {
    "MEDDL12M_A": {7, 8, 9},
    "MEDNG12M_A": {7, 8, 9},
    "AGEP_A": {97, 98, 99},
    "SEX_A": {7, 8, 9},
    "HISPALLP_A": {97, 98, 99},
    "EDUCP_A": {97, 98, 99},
    "MARSTAT_A": {9},
    "PCNTADLT_A": {8},
    "PCNTKIDS_A": {8},
    "HOUTENURE_A": {7, 8, 9},
    "EMPWRKLSW1_A": {7, 8, 9},
    "NATUSBORN_A": {7, 8, 9},
    "CITZNSTP_A": {7, 8, 9},
    "NOTCOV_A": {7, 8, 9},
    "USUALPL_A": {7, 8, 9},
    "PHSTAT_A": {7, 8, 9},
    "ANYDIFF_A": {9},
}

CATEGORY_LABELS = {
    "SEX_A": {1: "Male", 2: "Female"},
    "HISPALLP_A": {
        1: "Hispanic",
        2: "Non-Hispanic White",
        3: "Non-Hispanic Black/African American",
        4: "Non-Hispanic Asian",
        5: "Non-Hispanic AIAN",
        6: "Non-Hispanic AIAN and another group",
        7: "Other single and multiple races",
    },
    "EDUCP_A": {
        1: "Grade 0-11",
        2: "12th grade, no diploma",
        3: "GED or equivalent",
        4: "High school graduate",
        5: "Some college, no degree",
        6: "Associate degree, occupational/technical/vocational",
        7: "Associate degree, academic",
        8: "Bachelor's degree",
        9: "Master's degree",
        10: "Professional or doctoral degree",
    },
    "MARSTAT_A": {
        1: "Married, spouse present",
        2: "Married, spouse not present",
        3: "Married, spouse presence unknown",
        4: "Widowed",
        5: "Divorced",
        6: "Separated",
        7: "Never married",
        8: "Living with a partner",
    },
    "HOUTENURE_A": {1: "Owned or being bought", 2: "Rented", 3: "Other arrangement"},
    "EMPWRKLSW1_A": {1: "Yes", 2: "No"},
    "NATUSBORN_A": {1: "U.S. or U.S. territory born", 2: "Born outside U.S./territory"},
    "CITZNSTP_A": {1: "U.S. citizen", 2: "Not a U.S. citizen"},
    "NOTCOV_A": {1: "Not covered", 2: "Covered"},
    "USUALPL_A": {1: "One usual place", 2: "No usual place", 3: "More than one usual place"},
    "REGION": {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"},
    "URBRRL23": {
        1: "Large central metro",
        2: "Large fringe metro",
        3: "Medium and small metro",
        4: "Nonmetropolitan",
    },
    "PHSTAT_A": {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"},
    "ANYDIFF_A": {1: "Yes", 2: "No"},
}

VARIABLE_DICTIONARY = [
    ("MEDDL12M_A", "Primary outcome", "Delayed medical care due to cost, past 12 months", "target", "Yes=1, No=0; special response codes become missing"),
    ("MEDNG12M_A", "Secondary outcome", "Needed medical care but did not get it due to cost, past 12 months", "secondary_outcome", "Never used as a predictor of MEDDL12M_A"),
    ("AGEP_A", "Age", "Age of Sample Adult, top-coded", "core_predictor", "85 represents age 85+; 97-99 become missing"),
    ("SEX_A", "Sex", "Sex of Sample Adult", "core_predictor", "Categorical"),
    ("HISPALLP_A", "Race/ethnicity", "Single and multiple race groups with Hispanic origin", "core_predictor", "Public-use combined recode"),
    ("EDUCP_A", "Education", "Educational level of Sample Adult", "core_predictor", "Categorical"),
    ("MARSTAT_A", "Marital status", "Current marital status of Sample Adult", "core_predictor", "Code 9 unknown becomes missing"),
    ("PCNTADLT_A", "Family adults", "Number of adults in Sample Adult family", "core_predictor", "Top-coded at 3+"),
    ("PCNTKIDS_A", "Family children", "Number of children in Sample Adult family", "core_predictor", "Top-coded at 3+"),
    ("HOUTENURE_A", "Housing tenure", "Residence owned or rented", "core_predictor", "Categorical"),
    ("EMPWRKLSW1_A", "Employment", "Worked last week", "core_predictor", "Edited employment recode"),
    ("NATUSBORN_A", "Nativity", "Born in U.S. or U.S. territory", "core_predictor", "Categorical"),
    ("CITZNSTP_A", "Citizenship", "Citizenship status", "descriptive_sensitive", "Retained for subgroup analysis; excluded from default core model"),
    ("NOTCOV_A", "Insurance", "Coverage status as used in Health, United States", "core_predictor", "Not covered vs covered"),
    ("USUALPL_A", "Usual care", "Have a usual place for care", "core_predictor", "Categorical"),
    ("REGION", "Region", "Household region", "core_predictor", "Northeast/Midwest/South/West"),
    ("URBRRL23", "Urban-rural", "2023 NCHS Urban-Rural Classification Scheme for Counties", "core_predictor", "Four public-use categories"),
    ("PHSTAT_A", "General health", "General health status", "expanded_predictor", "Added only in expanded model"),
    ("ANYDIFF_A", "Functioning", "Functioning difficulties status", "expanded_predictor", "Added only in expanded model"),
    ("POVRATTC_A", "Poverty ratio", "Sample Adult family poverty ratio, top-coded", "income_predictor", "Analyze across all 10 income imputations"),
    ("RATCAT_A", "Poverty category", "Ratio of family income to poverty threshold", "income_predictor", "Analyze across all 10 income imputations"),
    ("WTFA_A", "Survey weight", "Final annual Sample Adult weight", "survey_design", "Use for population estimates"),
    ("PSTRAT", "Variance stratum", "Annual public variance stratum", "survey_design", "Use for survey variance estimation"),
    ("PPSU", "Variance PSU", "Annual public variance PSU", "survey_design", "Use for survey variance estimation"),
]

CORE_MODEL_COLUMNS = [
    "age",
    "sex",
    "race_ethnicity",
    "education",
    "marital_status",
    "family_adults",
    "family_children",
    "housing_tenure",
    "worked_last_week",
    "nativity",
    "insurance_status",
    "usual_place_care",
    "region",
    "urban_rural",
]

EXPANDED_MODEL_COLUMNS = CORE_MODEL_COLUMNS + ["general_health", "functioning_difficulty"]


def _require_files():
    missing = [str(p) for p in (ADULT_FILE, INCOME_FILE) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing raw NHIS file(s): " + ", ".join(missing))


def _recode_missing(series, variable):
    codes = MISSING_CODES.get(variable, set())
    return series.mask(series.isin(codes))


def _map_category(series, variable):
    cleaned = _recode_missing(series, variable)
    return cleaned.map(CATEGORY_LABELS[variable]).astype("string")


def _binary_yes_no(series, variable):
    cleaned = _recode_missing(series, variable)
    return cleaned.map({1: 1, 2: 0}).astype("Int64")


def load_source_data():
    _require_files()
    adult = pd.read_csv(ADULT_FILE, dtype={"HHX": "string"})
    income = pd.read_csv(INCOME_FILE, dtype={"HHX": "string"})
    return adult, income


def validate_source_data(adult, income):
    missing_cols = sorted(set(BASE_COLUMNS) - set(adult.columns))
    if missing_cols:
        raise ValueError(f"Adult file is missing expected columns: {missing_cols}")

    income_required = {"HHX", "IMPNUM_A", "POVRATTC_A", "RATCAT_A"}
    missing_income = sorted(income_required - set(income.columns))
    if missing_income:
        raise ValueError(f"Income file is missing expected columns: {missing_income}")

    if adult["HHX"].duplicated().any():
        raise ValueError("Adult file must contain one row per HHX.")

    counts = income.groupby("HHX", observed=True).size()
    if not (counts == 10).all():
        raise ValueError("Each Sample Adult must have exactly 10 income-imputation rows.")

    if set(income["IMPNUM_A"].dropna().astype(int).unique()) != set(range(1, 11)):
        raise ValueError("Income file must contain imputation numbers 1 through 10.")

    if set(adult["HHX"]) != set(income["HHX"]):
        raise ValueError("Adult and income files do not contain the same HHX identifiers.")


def build_adult_analysis(adult):
    out = adult[BASE_COLUMNS].copy()
    out["delayed_care_cost"] = _binary_yes_no(out["MEDDL12M_A"], "MEDDL12M_A")
    out["forgone_care_cost"] = _binary_yes_no(out["MEDNG12M_A"], "MEDNG12M_A")
    out["age"] = _recode_missing(out["AGEP_A"], "AGEP_A").astype("Float64")
    out["sex"] = _map_category(out["SEX_A"], "SEX_A")
    out["race_ethnicity"] = _map_category(out["HISPALLP_A"], "HISPALLP_A")
    out["education"] = _map_category(out["EDUCP_A"], "EDUCP_A")
    out["marital_status"] = _map_category(out["MARSTAT_A"], "MARSTAT_A")
    out["family_adults"] = _recode_missing(out["PCNTADLT_A"], "PCNTADLT_A").astype("Float64")
    out["family_children"] = _recode_missing(out["PCNTKIDS_A"], "PCNTKIDS_A").astype("Float64")
    out["housing_tenure"] = _map_category(out["HOUTENURE_A"], "HOUTENURE_A")
    out["worked_last_week"] = _map_category(out["EMPWRKLSW1_A"], "EMPWRKLSW1_A")
    out["nativity"] = _map_category(out["NATUSBORN_A"], "NATUSBORN_A")
    out["citizenship_status"] = _map_category(out["CITZNSTP_A"], "CITZNSTP_A")
    out["insurance_status"] = _map_category(out["NOTCOV_A"], "NOTCOV_A")
    out["usual_place_care"] = _map_category(out["USUALPL_A"], "USUALPL_A")
    out["region"] = _map_category(out["REGION"], "REGION")
    out["urban_rural"] = _map_category(out["URBRRL23"], "URBRRL23")
    out["general_health"] = _map_category(out["PHSTAT_A"], "PHSTAT_A")
    out["functioning_difficulty"] = _map_category(out["ANYDIFF_A"], "ANYDIFF_A")
    return out


def build_modeling_base(adult_analysis):
    cols = [
        "HHX",
        "delayed_care_cost",
        "WTFA_A",
        "PSTRAT",
        "PPSU",
        *EXPANDED_MODEL_COLUMNS,
    ]
    return adult_analysis[cols].copy()


def build_income_imputations(income):
    out = income[["HHX", "IMPNUM_A", "POVRATTC_A", "RATCAT_A"]].copy()
    out = out.sort_values(["HHX", "IMPNUM_A"]).reset_index(drop=True)
    return out


def build_variable_dictionary():
    return pd.DataFrame(
        VARIABLE_DICTIONARY,
        columns=["source_variable", "friendly_name", "description", "role", "treatment"],
    )


def weighted_prevalence(values, weights):
    mask = values.notna() & weights.notna()
    if not mask.any():
        return np.nan
    return float(np.average(values.loc[mask].astype(float), weights=weights.loc[mask].astype(float)))


def build_summary(adult_analysis, income_imputations):
    valid_target = adult_analysis["delayed_care_cost"].dropna()
    summary = {
        "adult_rows": int(len(adult_analysis)),
        "unique_adults": int(adult_analysis["HHX"].nunique()),
        "income_rows": int(len(income_imputations)),
        "income_imputations_per_adult": int(income_imputations.groupby("HHX").size().iloc[0]),
        "target_valid_rows": int(valid_target.shape[0]),
        "target_positive_rows": int((valid_target == 1).sum()),
        "target_negative_rows": int((valid_target == 0).sum()),
        "target_unweighted_prevalence": float(valid_target.mean()),
        "target_weighted_point_estimate": weighted_prevalence(
            adult_analysis["delayed_care_cost"], adult_analysis["WTFA_A"]
        ),
        "secondary_positive_rows": int((adult_analysis["forgone_care_cost"] == 1).sum()),
        "income_imputation_numbers": sorted(
            income_imputations["IMPNUM_A"].dropna().astype(int).unique().tolist()
        ),
        "note": "Weighted prevalence is a point estimate only; survey-design variance estimation is handled in the statistical-analysis stage.",
    }
    return summary


def prepare_data():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    adult, income = load_source_data()
    validate_source_data(adult, income)

    adult_analysis = build_adult_analysis(adult)
    modeling_base = build_modeling_base(adult_analysis)
    income_imputations = build_income_imputations(income)
    variable_dictionary = build_variable_dictionary()
    summary = build_summary(adult_analysis, income_imputations)

    adult_analysis.to_csv(PROCESSED_DIR / "adult_analysis.csv", index=False)
    modeling_base.to_csv(PROCESSED_DIR / "modeling_base.csv", index=False)
    income_imputations.to_csv(PROCESSED_DIR / "income_imputations.csv", index=False)
    variable_dictionary.to_csv(PROCESSED_DIR / "variable_dictionary.csv", index=False)
    (PROCESSED_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2) + "
")

    return summary


if __name__ == "__main__":
    result = prepare_data()
    print(json.dumps(result, indent=2))
