from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ASSETS = ROOT / "assets"


def _escape(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _write_calibration_svg():
    frame = pd.read_csv(PROCESSED / "boosting_calibration.csv")
    width = 760
    height = 460
    left = 76
    top = 54
    plot = 330
    scale = plot / 0.30

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="28" y="30" font-size="20" font-family="Arial" font-weight="700">Gradient-boosting calibration by held-out risk bin</text>',
        '<text x="28" y="50" font-size="12" font-family="Arial" fill="#5f6b76">Survey-weighted predicted probability vs. observed delayed-care prevalence</text>',
        f'<line x1="{left}" y1="{top + plot}" x2="{left + plot}" y2="{top}" stroke="#9aa5ad" stroke-width="1.5" stroke-dasharray="5 5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot}" x2="{left + plot}" y2="{top + plot}" stroke="#333" stroke-width="1"/>',
    ]

    for tick in [0.0, 0.1, 0.2, 0.3]:
        x = left + tick * scale
        y = top + plot - tick * scale
        parts.extend([
            f'<line x1="{x:.1f}" y1="{top + plot}" x2="{x:.1f}" y2="{top + plot + 5}" stroke="#333"/>',
            f'<text x="{x:.1f}" y="{top + plot + 22}" text-anchor="middle" font-size="11" font-family="Arial">{tick:.0%}</text>',
            f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#333"/>',
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11" font-family="Arial">{tick:.0%}</text>',
        ])

    for _, row in frame.iterrows():
        x = left + float(row["weighted_mean_predicted_probability"]) * scale
        y = top + plot - float(row["weighted_observed_prevalence"]) * scale
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="#315b7d"><title>Bin {int(row["risk_bin"])}: predicted {row["weighted_mean_predicted_probability"]:.1%}, observed {row["weighted_observed_prevalence"]:.1%}</title></circle>'
        )

    parts.extend([
        f'<text x="{left + plot / 2}" y="{top + plot + 48}" text-anchor="middle" font-size="12" font-family="Arial">Weighted mean predicted probability</text>',
        f'<text x="20" y="{top + plot / 2}" transform="rotate(-90 20 {top + plot / 2})" text-anchor="middle" font-size="12" font-family="Arial">Weighted observed prevalence</text>',
        '<text x="460" y="105" font-size="13" font-family="Arial" font-weight="700">Interpretation</text>',
        '<text x="460" y="128" font-size="12" font-family="Arial">Points near the dashed line indicate</text>',
        '<text x="460" y="146" font-size="12" font-family="Arial">agreement between predicted risk and</text>',
        '<text x="460" y="164" font-size="12" font-family="Arial">observed weighted prevalence.</text>',
        '<text x="460" y="205" font-size="12" font-family="Arial">Bins contain approximately equal</text>',
        '<text x="460" y="223" font-size="12" font-family="Arial">numbers of held-out respondents.</text>',
        '</svg>',
    ])
    (ASSETS / "calibration_diagnostics.svg").write_text("\n".join(parts), encoding="utf-8")


def _write_subgroup_svg():
    frame = pd.read_csv(PROCESSED / "model_subgroup_metrics.csv")
    rows = (
        frame.loc[:, ["dimension", "level", "model", "unweighted_n", "unweighted_positives", "weighted_roc_auc"]]
        .pivot(index=["dimension", "level", "unweighted_n", "unweighted_positives"], columns="model", values="weighted_roc_auc")
        .reset_index()
    )

    width = 940
    row_h = 34
    height = 100 + len(rows) * row_h + 50
    label_x = 30
    plot_left = 580
    plot_width = 300
    min_auc = 0.45
    max_auc = 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="28" y="30" font-size="20" font-family="Arial" font-weight="700">Held-out subgroup discrimination diagnostics</text>',
        '<text x="28" y="50" font-size="12" font-family="Arial" fill="#5f6b76">Weighted ROC-AUC; small subgroup counts are shown to discourage over-interpretation</text>',
        '<circle cx="600" cy="72" r="5" fill="#315b7d"/><text x="612" y="76" font-size="11" font-family="Arial">Expanded logistic</text>',
        '<circle cx="730" cy="72" r="5" fill="#7b4fa3"/><text x="742" y="76" font-size="11" font-family="Arial">Gradient boosting</text>',
    ]

    y = 104
    current_dimension = None
    for _, row in rows.iterrows():
        dimension = row["dimension"]
        if dimension != current_dimension:
            current_dimension = dimension
        label = f'{dimension} · {row["level"]}'
        count = f'n={int(row["unweighted_n"])}, positives={int(row["unweighted_positives"])}'
        parts.append(f'<text x="{label_x}" y="{y}" font-size="11" font-family="Arial">{_escape(label)}</text>')
        parts.append(f'<text x="360" y="{y}" font-size="10" font-family="Arial" fill="#6b7280">{count}</text>')
        parts.append(f'<line x1="{plot_left}" y1="{y - 4}" x2="{plot_left + plot_width}" y2="{y - 4}" stroke="#edf0f2"/>')

        for model, color in [
            ("expanded_logistic", "#315b7d"),
            ("expanded_gradient_boosting", "#7b4fa3"),
        ]:
            value = row.get(model)
            if pd.notna(value):
                x = plot_left + (float(value) - min_auc) / (max_auc - min_auc) * plot_width
                parts.append(f'<circle cx="{x:.1f}" cy="{y - 4}" r="4.8" fill="{color}"><title>{_escape(label)} · {model}: {float(value):.3f}</title></circle>')
        y += row_h

    for tick in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        x = plot_left + (tick - min_auc) / (max_auc - min_auc) * plot_width
        parts.append(f'<text x="{x:.1f}" y="{height - 18}" text-anchor="middle" font-size="10" font-family="Arial">{tick:.1f}</text>')
        parts.append(f'<line x1="{x:.1f}" y1="88" x2="{x:.1f}" y2="{height - 34}" stroke="#d9dee2" stroke-dasharray="2 4"/>')

    parts.append('</svg>')
    (ASSETS / "subgroup_model_diagnostics.svg").write_text("\n".join(parts), encoding="utf-8")


def build_review_artifacts():
    ASSETS.mkdir(parents=True, exist_ok=True)

    data_summary = json.loads((PROCESSED / "data_summary.json").read_text())
    survey_summary = json.loads((PROCESSED / "survey_summary.json").read_text())
    model_summary = json.loads((PROCESSED / "model_summary.json").read_text())
    groups = pd.read_csv(PROCESSED / "survey_group_estimates.csv")
    income = pd.read_csv(PROCESSED / "income_pooled_estimates.csv")
    metrics = pd.read_csv(PROCESSED / "model_metrics.csv").set_index("model")

    def group_value(dimension, level):
        row = groups.loc[(groups["dimension"] == dimension) & (groups["level"] == level)].iloc[0]
        return float(row["weighted_prevalence"])

    summary = {
        "sample": {
            "adult_rows": int(data_summary["adult_rows"]),
            "valid_primary_outcomes": int(data_summary["target_valid_rows"]),
            "positive_primary_outcomes": int(data_summary["target_positive_rows"]),
            "income_imputations": int(data_summary["income_imputations_per_adult"]),
        },
        "population_inference": {
            "overall_prevalence": float(survey_summary["overall_weighted_prevalence"]),
            "overall_ci_lower": float(survey_summary["overall_ci_lower"]),
            "overall_ci_upper": float(survey_summary["overall_ci_upper"]),
            "design_df": int(survey_summary["design_df"]),
            "not_covered_prevalence": group_value("Insurance status", "Not covered"),
            "covered_prevalence": group_value("Insurance status", "Covered"),
            "no_usual_place_prevalence": group_value("Usual place of care", "No usual place"),
            "poor_health_prevalence": group_value("General health", "Poor"),
        },
        "income_inference": {
            str(row["poverty_band"]): float(row["weighted_prevalence"])
            for _, row in income.iterrows()
        },
        "predictive_modeling": {
            "training_rows": int(model_summary["training_rows"]),
            "test_rows": int(model_summary["test_rows"]),
            "test_positive_rows": int(model_summary["test_positive_rows"]),
            "expanded_logistic_weighted_roc_auc": float(metrics.loc["expanded_logistic", "weighted_roc_auc"]),
            "expanded_logistic_weighted_average_precision": float(metrics.loc["expanded_logistic", "weighted_average_precision"]),
            "boosting_weighted_roc_auc": float(metrics.loc["expanded_gradient_boosting", "weighted_roc_auc"]),
            "boosting_weighted_average_precision": float(metrics.loc["expanded_gradient_boosting", "weighted_average_precision"]),
            "boosting_weighted_brier_score": float(metrics.loc["expanded_gradient_boosting", "weighted_brier_score"]),
            "boosting_top_decile_positive_capture": float(model_summary["boosting_top_decile_positive_capture"]),
        },
        "interpretation": {
            "design": "Cross-sectional NHIS analysis; population estimates are survey-weighted and predictive models are retrospective, not prospective or causal.",
            "validation": "Representative survey point estimates and standard errors are independently cross-checked against R survey in CI.",
        },
    }

    (PROCESSED / "project_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_calibration_svg()
    _write_subgroup_svg()
    return summary


if __name__ == "__main__":
    build_review_artifacts()
