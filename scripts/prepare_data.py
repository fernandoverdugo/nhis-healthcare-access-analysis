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
    return adult_analysis%¶6öÇ5Òæ6÷’‚  ¦FVb'V–ÆEö–æ6öÖUö–×WFF–öç2†–æ6öÖR“ ¢÷WBÒ–æ6öÖUµ²$„…‚"Â$”ÕåTÕô"Â%õe$ED5ô"Â%$D4Eô%ÕÒæ6÷’‚¢÷WBÒ÷WBç6÷'E÷fÇVW2…²$„…‚"Â$”ÕåTÕô%Ò’ç&W6WEö–æFW‚†G&÷ÕG'VR¢&WGW&â÷W@  ¦FVb'V–ÆE÷f&–&ÆUöF–7F–öæ'’‚“ ¢&WGW&âBäFFg&ÖR€¢d$”$ÄUôD”5D”ôä%’À¢6öÇVÖç3Õ²'6÷W&6U÷f&–&ÆR"Â&g&–VæFÇ•öæÖR"Â&FW67&—F–öâ"Â'&öÆR"Â'G&VFÖVçB%ÒÀ¢  ¦FVbvV–v‡FVE÷&WfÆVæ6R‡fÇVW2ÂvV–v‡G2“ ¢Ö6²ÒfÇVW2ææ÷Fæ‚’bvV–v‡G2ææ÷Fæ‚¢–bæ÷BÖ6²æç’‚“ ¢&WGW&âçææà¢&WGW&âfÆöB†çæfW&vR‡fÇVW2æÆö5¶Ö6µÒæ7G—R†fÆöB’ÂvV–v‡G3×vV–v‡G2æÆö5¶Ö6µÒæ7G—R†fÆöB’’  ¦FVb'V–ÆE÷7VÖÖ'’†GVÇEöæÇ—6—2Â–æ6öÖUö–×WFF–öç2“ ¢fÆ–E÷F&vWBÒGVÇEöæÇ—6—5²&FVÆ–VEö6&Uö6÷7B%ÒæG&÷æ‚¢7VÖÖ'’Ò°¢&GVÇE÷&÷w2#¢–çB†ÆVâ†GVÇEöæÇ—6—2’’À¢'Væ—VUöGVÇG2#¢–çB†GVÇEöæÇ—6—5²$„…‚%ÒæçVæ—VR‚’’À¢&–æ6öÖU÷&÷w2#¢–çB†ÆVâ†–æ6öÖUö–×WFF–öç2’’À¢&–æ6öÖUö–×WFF–öç5÷W%öGVÇB#¢–çB†–æ6öÖUö–×WFF–öç2æw&÷W'’‚$„…‚"Âö'6W'fVCÕG'VR’ç6—¦R‚’æ–Æö5³Ò’À¢'F&vWE÷fÆ–E÷&÷w2#¢–çB‡fÆ–E÷F&vWBç6†U³Ò’À¢'F&vWE÷÷6—F—fU÷&÷w2#¢–çB‚‡fÆ–E÷F&vWBÓÒ’ç7VÒ‚’’À¢'F&vWEöæVvF—fU÷&÷w2#¢–çB‚‡fÆ–E÷F&vWBÓÒ’ç7VÒ‚’’À¢'F&vWE÷VçvV–v‡FVE÷&WfÆVæ6R#¢fÆöB‡fÆ–E÷F&vWBæÖVâ‚’’À¢'F&vWE÷vV–v‡FVE÷ö–çEöW7F–ÖFR#¢vV–v‡FVE÷&WfÆVæ6R€¢GVÇEöæÇ—6—5²&FVÆ–VEö6&Uö6÷7B%ÒÂGVÇEöæÇ—6—5²%uDdô%Ğ¢’À¢'6V6öæF'•÷÷6—F—fU÷&÷w2#¢–çB‚†GVÇEöæÇ—6—5²&f÷&vöæUö6&Uö6÷7B%ÒÓÒ’ç7VÒ‚’’À¢&–æ6öÖUö–×WFF–öåöçVÖ&W'2#¢6÷'FVB€¢–æ6öÖUö–×WFF–öç5²$”ÕåTÕô%ÒæG&÷æ‚’æ7G—R†–çB’çVæ—VR‚’çFöÆ—7B‚¢’À¢&æ÷FR#¢%vV–v‡FVB&WfÆVæ6R—2ö–çBW7F–ÖFRöæÇ“²7W'fW’ÖFW6–vâf&–æ6RW7F–ÖF–öâ—2†æFÆVB–âF†R7FF—7F–6ÂÖæÇ—6—27FvRâ"À¢Ğ¢&WGW&â7VÖÖ'  ¦FVb&W&UöFF‚“ ¢$ô4U54TEôD•"æÖ¶F—"‡&VçG3ÕG'VRÂW†—7Eöö³ÕG'VR¢GVÇBÂ–æ6öÖRÒÆöE÷6÷W&6UöFF‚¢fÆ–FFU÷6÷W&6UöFF†GVÇBÂ–æ6öÖR ¢GVÇEöæÇ—6—2Ò'V–ÆEöGVÇEöæÇ—6—2†GVÇB¢ÖöFVÆ–æuö&6RÒ'V–ÆEöÖöFVÆ–æuö&6R†GVÇEöæÇ—6—2¢–æ6öÖUö–×WFF–öç2Ò'V–ÆEö–æ6öÖUö–×WFF–öç2†–æ6öÖR¢f&–&ÆUöF–7F–öæ'’Ò'V–ÆE÷f&–&ÆUöF–7F–öæ'’‚¢7VÖÖ'’Ò'V–ÆE÷7VÖÖ'’†GVÇEöæÇ—6—2Â–æ6öÖUö–×WFF–öç2 ¢GVÇEöæÇ—6—2çFõö77b…$ô4U54TEôD•"ò&GVÇEöæÇ—6—2æ77b"Â–æFWƒÔfÇ6R¢ÖöFVÆ–æuö&6RçFõö77b…$ô4U54TEôD•"ò&ÖöFVÆ–æuö&6Ræ77b"Â–æFWƒÔfÇ6R¢–æ6öÖUö–×WFF–öç2çFõö77b…$ô4U54TEôD•"ò&–æ6öÖUö–×WFF–öç2æ77b"Â–æFWƒÔfÇ6R¢f&–&ÆUöF–7F–öæ'’çFõö77b…$ô4U54TEôD•"ò'f&–&ÆUöF–7F–öæ'’æ77b"Â–æFWƒÔfÇ6R¢…$ô4U54TEôD•"ò&FF÷7VÖÖ'’æ§6öâ"’çw&—FU÷FW‡B†§6öâæGV×2‡7VÖÖ'’Â–æFVçCÓ"’²%Æâ" ¢&WGW&â7VÖÖ'  ¦–bõöæÖUõòÓÒ%õöÖ–åõò# ¢&W7VÇBÒ&W&UöFF‚¢&–çB†§6öâæGV×2‡&W7VÇBÂ–æFVçCÓ"’ 