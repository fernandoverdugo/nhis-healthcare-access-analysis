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
