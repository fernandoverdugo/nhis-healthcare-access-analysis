from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ASSETS = ROOT / "assets"
NOTEBOOKS = ROOT / "notebooks"


def test_committed_notebooks_are_fully_executed_without_errors():
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        notebook = json.loads(path.read_text())
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert code_cells
        assert all(cell.get("execution_count") is not None for cell in code_cells)
        errors = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        assert errors == []


def test_readme_model_svg_matches_current_metrics():
    metrics = pd.read_csv(PROCESSED / "model_metrics.csv").set_index("model")
    svg = (ASSETS / "model_discrimination_metrics.svg").read_text()
    expected = [
        f"{metrics.loc['prevalence_baseline', 'weighted_roc_auc']:.1%}",
        f"{metrics.loc['core_logistic', 'weighted_roc_auc']:.1%}",
        f"{metrics.loc['expanded_logistic', 'weighted_roc_auc']:.1%}",
        f"{metrics.loc['expanded_gradient_boosting', 'weighted_roc_auc']:.1%}",
    ]
    for label in expected:
        assert label in svg


def test_readme_income_svg_matches_current_estimates():
    income = pd.read_csv(PROCESSED / "income_pooled_estimates.csv")
    svg = (ASSETS / "income_prevalence.svg").read_text()
    for value in income["weighted_prevalence"]:
        assert f"{value:.1%}" in svg


def test_readme_risk_svg_matches_current_concentration():
    deciles = pd.read_csv(PROCESSED / "boosting_risk_deciles.csv")
    svg = (ASSETS / "risk_concentration.svg").read_text()
    assert f"{deciles.loc[0, 'weighted_positive_share']:.1%}" in svg
    assert f"{deciles.loc[1, 'cumulative_weighted_positive_share']:.1%}" in svg


def test_calibration_svg_matches_current_bins():
    calibration = pd.read_csv(PROCESSED / "boosting_calibration.csv")
    svg = (ASSETS / "calibration_diagnostics.svg").read_text()
    for _, row in calibration.iterrows():
        assert f"predicted {row['weighted_mean_predicted_probability']:.1%}" in svg
        assert f"observed {row['weighted_observed_prevalence']:.1%}" in svg


def test_subgroup_svg_matches_current_diagnostics():
    subgroup = pd.read_csv(PROCESSED / "model_subgroup_metrics.csv")
    svg = (ASSETS / "subgroup_model_diagnostics.svg").read_text()
    for dimension in ["Sex", "Race/ethnicity", "Nativity"]:
        assert dimension in svg
    for _, row in subgroup.iterrows():
        assert str(row["level"]) in svg
        assert f"n={int(row['unweighted_n'])}, positives={int(row['unweighted_positives'])}" in svg


def test_project_summary_matches_current_outputs():
    summary = json.loads((PROCESSED / "project_summary.json").read_text())
    overall = pd.read_csv(PROCESSED / "survey_overall.csv").iloc[0]
    metrics = pd.read_csv(PROCESSED / "model_metrics.csv").set_index("model")
    deciles = pd.read_csv(PROCESSED / "boosting_risk_deciles.csv")

    assert abs(summary["population_inference"]["overall_prevalence"] - overall["weighted_prevalence"]) < 1e-12
    assert abs(
        summary["predictive_modeling"]["boosting_weighted_roc_auc"]
        - metrics.loc["expanded_gradient_boosting", "weighted_roc_auc"]
    ) < 1e-12
    assert abs(
        summary["predictive_modeling"]["boosting_top_decile_positive_capture"]
        - deciles.loc[0, "weighted_positive_share"]
    ) < 1e-12
