from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

ADULT_ANALYSIS_FILE = PROCESSED_DIR / "adult_analysis.csv"
INCOME_IMPUTATIONS_FILE = PROCESSED_DIR / "income_imputations.csv"

OUTCOME = "delayed_care_cost"
WEIGHT = "WTFA_A"
STRATA = "PSTRAT"
PSU = "PPSU"

GROUP_SPECS = {
    "Insurance status": "insurance_status",
    "Usual place of care": "usual_place_care",
    "Education": "education_group",
    "Region": "region",
    "General health": "general_health",
    "Functioning difficulty": "functioning_difficulty",
    "Nativity": "nativity",
    "Urban-rural classification": "urban_rural",
    "Housing tenure": "housing_tenure",
}

EDUCATION_GROUPS = {
    "Grade 0-11": "Less than high school",
    "12th grade, no diploma": "Less than high school",
    "GED or equivalent": "High school/GED",
    "High school graduate": "High school/GED",
    "Some college, no degree": "Some college/associate",
    "Associate degree, occupational/technical/vocational": "Some college/associate",
    "Associate degree, academic": "Some college/associate",
    "Bachelor's degree": "Bachelor's or higher",
    "Master's degree": "Bachelor's or higher",
    "Professional or doctoral degree": "Bachelor's or higher",
}

POVERTY_BANDS = {
    1: "Below 100% FPL",
    2: "Below 100% FPL",
    3: "Below 100% FPL",
    4: "100-199% FPL",
    5: "100-199% FPL",
    6: "100-199% FPL",
    7: "100-199% FPL",
    8: "200-399% FPL",
    9: "200-399% FPL",
    10: "200-399% FPL",
    11: "200-399% FPL",
    12: "400%+ FPL",
    13: "400%+ FPL",
    14: "400%+ FPL",
}

POVERTY_ORDER = ["Below 100% FPL", "100-199% FPL", "200-399% FPL", "400%+ FPL"]


def _load_data():
    adult = pd.read_csv(ADULT_ANALYSIS_FILE, dtype={"HHX": "string"})
    income = pd.read_csv(INCOME_IMPUTATIONS_FILE, dtype={"HHX": "string"})
    adult["education_group"] = adult["education"].map(EDUCATION_GROUPS)
    return adult, income


def design_degrees_of_freedom(df):
    design = df[[STRATA, PSU]].dropna().drop_duplicates()
    return int(len(design) - design[STRATA].nunique())


def survey_binary_mean(df, domain=None, alpha=0.05):
    cols = [OUTCOME, WEIGHT, STRATA, PSU]
    work = df[cols].copy()
    work = work.dropna(subset=cols)

    if domain is None:
        domain_values = pd.Series(1.0, index=work.index)
    else:
        domain_values = domain.reindex(work.index).fillna(False).astype(float)

    y = work[OUTCOME].astype(float)
    w = work[WEIGHT].astype(float)
    denominator = float((w * domain_values).sum())
    if denominator <= 0:
        return None

    numerator = float((w * domain_values * y).sum())
    estimate = numerator / denominator

    linearized = w * domain_values * (y - estimate) / denominator
    lin_df = pd.DataFrame({STRATA: work[STRATA], PSU: work[PSU], "linearized": linearized})
    psu_totals = lin_df.groupby([STRATA, PSU], observed=True)["linearized"].sum().reset_index()

    variance = 0.0
    for _, grp in psu_totals.groupby(STRATA, observed=True):
        n_psu = len(grp)
        if n_psu < 2:
            continue
        values = grp["linearized"].to_numpy(dtype=float)
        centered = values - values.mean()
        variance += n_psu / (n_psu - 1) * float(np.square(centered).sum())

    se = float(np.sqrt(max(variance, 0.0)))
    df_design = design_degrees_of_freedom(work)
    critical = float(t.ppf(1 - alpha / 2, df_design))
    lower = max(0.0, estimate - critical * se)
    upper = min(1.0, estimate + critical * se)

    domain_bool = domain_values.astype(bool)
    n_unweighted = int(domain_bool.sum())
    positives = int(y.loc[domain_bool].sum())

    return {
        "unweighted_n": n_unweighted,
        "unweighted_positives": positives,
        "weighted_prevalence": estimate,
        "standard_error": se,
        "ci_lower": lower,
        "ci_upper": upper,
        "design_df": df_design,
    }


def build_overall_estimate(adult):
    result = survey_binary_mean(adult)
    return pd.DataFrame([{"dimension": "Overall", "level": "All adults", **result}])


def build_group_estimates(adult):
    rows = []
    for dimension, column in GROUP_SPECS.items():
        levels = adult[column].dropna().drop_duplicates().tolist()
        for level in levels:
            domain = adult[column].eq(level)
            result = survey_binary_mean(adult, domain=domain)
            if result is not None and result["unweighted_n"] > 0:
                rows.append({"dimension": dimension, "level": str(level), **result})
    return pd.DataFrame(rows)


def _rubin_pool(estimates, variances, complete_df):
    q = np.asarray(estimates, dtype=float)
    u = np.asarray(variances, dtype=float)
    m = len(q)
    qbar = float(q.mean())
    ubar = float(u.mean())
    between = float(q.var(ddof=1)) if m > 1 else 0.0
    total = ubar + (1 + 1 / m) * between

    if total <= 0:
        gamma = 0.0
    else:
        gamma = float(((1 + 1 / m) * between) / total)

    if gamma <= 1e-18:
        df_old = float("inf")
    else:
        df_old = float((m - 1) / (gamma ** 2))

    df_observed = float(complete_df * (complete_df + 1) / (complete_df + 3) * (1 - gamma))
    if df_observed <= 0:
        df_mi = df_old
    elif np.isinf(df_old):
        df_mi = df_observed
    else:
        df_mi = float(1 / (1 / df_old + 1 / df_observed))

    critical = float(t.ppf(0.975, df_mi))
    se = float(np.sqrt(max(total, 0.0)))
    return {
        "weighted_prevalence": qbar,
        "standard_error": se,
        "ci_lower": max(0.0, qbar - critical * se),
        "ci_upper": min(1.0, qbar + critical * se),
        "within_imputation_variance": ubar,
        "between_imputation_variance": between,
        "total_variance": total,
        "fraction_missing_information": gamma,
        "mi_df": df_mi,
        "imputations": m,
    }


def build_income_pooled_estimates(adult, income):
    target_cols = ["HHX", OUTCOME, WEIGHT, STRATA, PSU]
    base = adult[target_cols].copy()
    complete_df = design_degrees_of_freedom(base.dropna(subset=[OUTCOME, WEIGHT, STRATA, PSU]))

    by_band = {band: [] for band in POVERTY_ORDER}
    n_by_band = {band: [] for band in POVERTY_ORDER}
    positive_by_band = {band: [] for band in POVERTY_ORDER}

    for imp in range(1, 11):
        imp_df = income.loc[income["IMPNUM_A"] == imp, ["HHX", "RATCAT_A"]].copy()
        imp_df["poverty_band"] = imp_df["RATCAT_A"].map(POVERTY_BANDS)
        merged = base.merge(imp_df[["HHX", "poverty_band"]], on="HHX", how="left", validate="one_to_one")

        for band in POVERTY_ORDER:
            result = survey_binary_mean(merged, domain=merged["poverty_band"].eq(band))
            by_band[band].append((result["weighted_prevalence"], result["standard_error"] ** 2))
            n_by_band[band].append(result["unweighted_n"])
            positive_by_band[band].append(result["unweighted_positives"])

    rows = []
    for band in POVERTY_ORDER:
        vals = by_band[band]
        pooled = _rubin_pool([x[0] for x in vals], [x[1] for x in vals], complete_df)
        rows.append(
            {
                "poverty_band": band,
                "mean_unweighted_n": float(np.mean(n_by_band[band])),
                "mean_unweighted_positives": float(np.mean(positive_by_band[band])),
                **pooled,
            }
        )
    return pd.DataFrame(rows)


def run_survey_analysis():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    adult, income = _load_data()

    overall = build_overall_estimate(adult)
    groups = build_group_estimates(adult)
    income_pooled = build_income_pooled_estimates(adult, income)

    overall.to_csv(PROCESSED_DIR / "survey_overall.csv", index=False)
    groups.to_csv(PROCESSED_DIR / "survey_group_estimates.csv", index=False)
    income_pooled.to_csv(PROCESSED_DIR / "income_pooled_estimates.csv", index=False)

    summary = {
        "overall_weighted_prevalence": float(overall.loc[0, "weighted_prevalence"]),
        "overall_ci_lower": float(overall.loc[0, "ci_lower"]),
        "overall_ci_upper": float(overall.loc[0, "ci_upper"]),
        "design_df": int(overall.loc[0, "design_df"]),
        "income_imputations": 10,
        "income_pooling": "Rubin pooling of survey-weighted point estimates and Taylor-linearized within-imputation variances",
    }
    (PROCESSED_DIR / "survey_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_survey_analysis(), indent=2))
