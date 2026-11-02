from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from survey_analysis import _load_data as load_survey_data, survey_binary_mean
from train_models import _load_data as load_model_data, _make_split

PROCESSED = ROOT / "data" / "processed"


def test_survey_overall_estimate_and_ci():
    adult, _ = load_survey_data()
    result = survey_binary_mean(adult)
    assert 0.06 < result["weighted_prevalence"] < 0.08
    assert result["ci_lower"] < result["weighted_prevalence"] < result["ci_upper"]
    assert result["design_df"] == 562


def test_income_pooling_uses_all_ten_imputations():
    pooled = pd.read_csv(PROCESSED / "income_pooled_estimates.csv")
    assert len(pooled) == 4
    assert (pooled["imputations"] == 10).all()
    assert pooled["weighted_prevalence"].between(0, 1).all()


def test_higher_income_band_has_lower_delayed_care_prevalence():
    pooled = pd.read_csv(PROCESSED / "income_pooled_estimates.csv").set_index("poverty_band")
    assert pooled.loc["400%+ FPL", "weighted_prevalence"] < pooled.loc["Below 100% FPL", "weighted_prevalence"]
    assert pooled.loc["400%+ FPL", "weighted_prevalence"] < pooled.loc["100-199% FPL", "weighted_prevalence"]


def test_model_split_is_disjoint_and_complete():
    model, _ = load_model_data()
    train, test, split = _make_split(model)
    assert set(train["HHX"]).isdisjoint(set(test["HHX"]))
    assert len(train) + len(test) == len(split)
    assert set(split["split"]) == {"train", "test"}


def test_generated_model_metrics_are_valid():
    metrics = pd.read_csv(PROCESSED / "model_metrics.csv").set_index("model")
    assert metrics["weighted_roc_auc"].between(0, 1).all()
    assert metrics["weighted_average_precision"].between(0, 1).all()
    assert metrics.loc["core_logistic", "weighted_roc_auc"] > metrics.loc["prevalence_baseline", "weighted_roc_auc"]
    assert metrics.loc["expanded_logistic", "weighted_roc_auc"] >= metrics.loc["core_logistic", "weighted_roc_auc"]


def test_income_model_runs_cover_all_imputations():
    runs = pd.read_csv(PROCESSED / "income_model_runs.csv")
    assert set(runs["model_family"]) == {"core_plus_income_logistic", "expanded_plus_income_logistic"}
    for _, grp in runs.groupby("model_family"):
        assert set(grp["imputation"]) == set(range(1, 11))


def test_gradient_boosting_benchmark_outputs_are_valid():
    metrics = pd.read_csv(PROCESSED / "model_metrics.csv").set_index("model")
    assert "expanded_gradient_boosting" in metrics.index
    assert 0.5 < metrics.loc["expanded_gradient_boosting", "weighted_roc_auc"] < 1
    assert metrics.loc["expanded_gradient_boosting", "weighted_brier_score"] < metrics.loc["prevalence_baseline", "weighted_brier_score"]


def test_boosting_permutation_importance_covers_expanded_features():
    importance = pd.read_csv(PROCESSED / "boosting_permutation_importance.csv")
    assert set(importance["feature"]) == {
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
        "general_health",
        "functioning_difficulty",
    }
    assert importance["importance_mean"].notna().all()
    assert importance.iloc[0]["importance_mean"] > 0


def test_boosting_calibration_bins_are_complete():
    calibration = pd.read_csv(PROCESSED / "boosting_calibration.csv")
    assert calibration["risk_bin"].tolist() == list(range(1, 11))
    assert calibration["weighted_mean_predicted_probability"].between(0, 1).all()
    assert calibration["weighted_observed_prevalence"].between(0, 1).all()


def test_boosting_risk_deciles_concentrate_positive_outcomes():
    deciles = pd.read_csv(PROCESSED / "boosting_risk_deciles.csv")
    assert deciles["risk_decile"].tolist() == list(range(1, 11))
    assert abs(deciles["weighted_positive_share"].sum() - 1.0) < 1e-9
    assert deciles.loc[0, "weighted_positive_share"] > 0.30
    assert deciles.loc[1, "cumulative_weighted_positive_share"] > 0.50


def test_income_pooling_reports_barnard_rubin_diagnostics():
    pooled = pd.read_csv(PROCESSED / "income_pooled_estimates.csv")
    assert "fraction_missing_information" in pooled.columns
    assert pooled["fraction_missing_information"].between(0, 1).all()
    assert pooled["mi_df"].gt(0).all()
    assert pooled["mi_df"].lt(562).all()
