from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prepare_data import CORE_MODEL_COLUMNS, EXPANDED_MODEL_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

MODEL_FILE = PROCESSED_DIR / "modeling_base.csv"
INCOME_FILE = PROCESSED_DIR / "income_imputations.csv"
RANDOM_STATE = 2026
TEST_SIZE = 0.25

BASE_NUMERIC = ["age", "family_adults", "family_children"]


def _load_data():
    model = pd.read_csv(MODEL_FILE, dtype={"HHX": "string"})
    income = pd.read_csv(INCOME_FILE, dtype={"HHX": "string"})
    return model, income


def _build_pipeline(features):
    numeric = [c for c in features if c in BASE_NUMERIC or c == "poverty_ratio"]
    categorical = [c for c in features if c not in numeric]

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )

    model = LogisticRegression(max_iter=300, solver="newton-cholesky", random_state=RANDOM_STATE)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])




def _build_boosting_pipeline(features):
    numeric = [c for c in features if c in BASE_NUMERIC]
    categorical = [c for c in features if c not in numeric]

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                drop="first",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _fit_boosting_model(train, test, features, model_name):
    pipeline = _build_boosting_pipeline(features)
    train_weights = _normalized_weights(train["WTFA_A"])
    pipeline.fit(
        train[features],
        train["delayed_care_cost"].astype(int),
        model__sample_weight=train_weights,
    )
    probabilities = pipeline.predict_proba(test[features])[:, 1]
    metrics = _metrics(
        test["delayed_care_cost"], probabilities, test["WTFA_A"], model_name
    )
    return pipeline, probabilities, metrics


def _weighted_calibration_table(y_true, probabilities, weights, bins=10):
    table = pd.DataFrame(
        {
            "outcome": np.asarray(y_true, dtype=int),
            "probability": np.asarray(probabilities, dtype=float),
            "weight": np.asarray(weights, dtype=float),
        }
    )
    table["risk_bin"] = pd.qcut(
        table["probability"].rank(method="first"), bins, labels=False
    ) + 1
    rows = []
    for risk_bin, grp in table.groupby("risk_bin", observed=True):
        rows.append(
            {
                "risk_bin": int(risk_bin),
                "rows": int(len(grp)),
                "weighted_mean_predicted_probability": float(
                    np.average(grp["probability"], weights=grp["weight"])
                ),
                "weighted_observed_prevalence": float(
                    np.average(grp["outcome"], weights=grp["weight"])
                ),
            }
        )
    return pd.DataFrame(rows)


def _risk_decile_table(y_true, probabilities, weights):
    table = pd.DataFrame(
        {
            "outcome": np.asarray(y_true, dtype=int),
            "probability": np.asarray(probabilities, dtype=float),
            "weight": np.asarray(weights, dtype=float),
        }
    ).sort_values("probability", ascending=False).reset_index(drop=True)
    table["risk_decile"] = pd.qcut(
        table.index.to_series().rank(method="first"), 10, labels=False
    ) + 1
    total_weighted_positives = float((table["outcome"] * table["weight"]).sum())
    rows = []
    cumulative_positive_weight = 0.0
    for decile, grp in table.groupby("risk_decile", sort=True, observed=True):
        positive_weight = float((grp["outcome"] * grp["weight"]).sum())
        cumulative_positive_weight += positive_weight
        rows.append(
            {
                "risk_decile": int(decile),
                "rows": int(len(grp)),
                "weighted_mean_predicted_probability": float(
                    np.average(grp["probability"], weights=grp["weight"])
                ),
                "weighted_observed_prevalence": float(
                    np.average(grp["outcome"], weights=grp["weight"])
                ),
                "weighted_positive_share": (
                    positive_weight / total_weighted_positives
                    if total_weighted_positives > 0
                    else np.nan
                ),
                "cumulative_weighted_positive_share": (
                    cumulative_positive_weight / total_weighted_positives
                    if total_weighted_positives > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)

def _normalized_weights(weights):
    weights = np.asarray(weights, dtype=float)
    return weights / weights.mean()


def _metrics(y_true, probabilities, weights, model_name):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    weights = np.asarray(weights, dtype=float)
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "model": model_name,
        "weighted_roc_auc": roc_auc_score(y_true, probabilities, sample_weight=weights),
        "weighted_average_precision": average_precision_score(y_true, probabilities, sample_weight=weights),
        "weighted_brier_score": brier_score_loss(y_true, probabilities, sample_weight=weights),
        "weighted_log_loss": log_loss(y_true, probabilities, sample_weight=weights, labels=[0, 1]),
        "weighted_precision_at_0_5": precision_score(
            y_true, predictions, sample_weight=weights, zero_division=0
        ),
        "weighted_recall_at_0_5": recall_score(
            y_true, predictions, sample_weight=weights, zero_division=0
        ),
        "weighted_specificity_at_0_5": recall_score(
            y_true, predictions, sample_weight=weights, pos_label=0, zero_division=0
        ),
        "weighted_f1_at_0_5": f1_score(y_true, predictions, sample_weight=weights, zero_division=0),
        "weighted_balanced_accuracy_at_0_5": balanced_accuracy_score(
            y_true, predictions, sample_weight=weights
        ),
        "unweighted_roc_auc": roc_auc_score(y_true, probabilities),
        "unweighted_average_precision": average_precision_score(y_true, probabilities),
    }


def _fit_model(train, test, features, model_name):
    pipeline = _build_pipeline(features)
    train_weights = _normalized_weights(train["WTFA_A"])
    pipeline.fit(train[features], train["delayed_care_cost"].astype(int), model__sample_weight=train_weights)
    probabilities = pipeline.predict_proba(test[features])[:, 1]
    metrics = _metrics(test["delayed_care_cost"], probabilities, test["WTFA_A"], model_name)
    return pipeline, probabilities, metrics


def _coefficient_table(pipeline, model_name):
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    table = pd.DataFrame({"feature": names, "coefficient": coefs})
    table["odds_ratio"] = np.exp(table["coefficient"])
    table["absolute_coefficient"] = table["coefficient"].abs()
    table["model"] = model_name
    return table.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)


def _make_split(model):
    eligible = model.dropna(subset=["delayed_care_cost", "WTFA_A"]).copy()
    train_ids, test_ids = train_test_split(
        eligible["HHX"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=eligible["delayed_care_cost"],
    )
    train_set = set(train_ids)
    split = eligible[["HHX"]].copy()
    split["split"] = np.where(split["HHX"].isin(train_set), "train", "test")
    train = eligible.loc[eligible["HHX"].isin(train_set)].copy()
    test = eligible.loc[~eligible["HHX"].isin(train_set)].copy()
    return train, test, split


def _income_model_runs(model, income, split, base_features, model_name):
    rows = []
    for imp in range(1, 11):
        imp_income = income.loc[income["IMPNUM_A"] == imp, ["HHX", "POVRATTC_A"]].rename(
            columns={"POVRATTC_A": "poverty_ratio"}
        )
        merged = model.merge(imp_income, on="HHX", how="left", validate="one_to_one")
        merged = merged.merge(split, on="HHX", how="inner", validate="one_to_one")
        train = merged.loc[merged["split"] == "train"].copy()
        test = merged.loc[merged["split"] == "test"].copy()
        features = [*base_features, "poverty_ratio"]
        _, _, metrics = _fit_model(train, test, features, f"{model_name}_imp{imp}")
        metrics["model_family"] = model_name
        metrics["imputation"] = imp
        rows.append(metrics)
    return pd.DataFrame(rows)


def _summarize_income_metrics(runs):
    metric_cols = [
        c
        for c in runs.columns
        if c.startswith("weighted_") or c.startswith("unweighted_")
    ]
    rows = []
    for model_family, grp in runs.groupby("model_family", observed=True):
        row = {"model": model_family, "imputations": len(grp)}
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(grp[metric].mean())
            row[f"{metric}_sd"] = float(grp[metric].std(ddof=1))
            row[f"{metric}_min"] = float(grp[metric].min())
            row[f"{metric}_max"] = float(grp[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def run_modeling():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    model, income = _load_data()
    train, test, split = _make_split(model)

    prevalence = np.average(train["delayed_care_cost"].astype(float), weights=train["WTFA_A"].astype(float))
    baseline_prob = np.full(len(test), prevalence, dtype=float)
    baseline_metrics = _metrics(
        test["delayed_care_cost"], baseline_prob, test["WTFA_A"], "prevalence_baseline"
    )

    core_pipe, core_prob, core_metrics = _fit_model(train, test, CORE_MODEL_COLUMNS, "core_logistic")
    expanded_pipe, expanded_prob, expanded_metrics = _fit_model(
        train, test, EXPANDED_MODEL_COLUMNS, "expanded_logistic"
    )
    boosting_pipe, boosting_prob, boosting_metrics = _fit_boosting_model(
        train, test, EXPANDED_MODEL_COLUMNS, "expanded_gradient_boosting"
    )

    metrics = pd.DataFrame(
        [baseline_metrics, core_metrics, expanded_metrics, boosting_metrics]
    )
    core_coef = _coefficient_table(core_pipe, "core_logistic")
    expanded_coef = _coefficient_table(expanded_pipe, "expanded_logistic")

    predictions = test[["HHX", "delayed_care_cost", "WTFA_A"]].copy()
    predictions["baseline_probability"] = baseline_prob
    predictions["core_probability"] = core_prob
    predictions["expanded_probability"] = expanded_prob
    predictions["boosting_probability"] = boosting_prob

    permutation = permutation_importance(
        boosting_pipe,
        test[EXPANDED_MODEL_COLUMNS],
        test["delayed_care_cost"].astype(int),
        scoring="roc_auc",
        n_repeats=5,
        random_state=RANDOM_STATE,
        sample_weight=test["WTFA_A"].astype(float),
        n_jobs=1,
    )
    boosting_importance = pd.DataFrame(
        {
            "feature": EXPANDED_MODEL_COLUMNS,
            "importance_mean": permutation.importances_mean,
            "importance_sd": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)

    boosting_calibration = _weighted_calibration_table(
        test["delayed_care_cost"], boosting_prob, test["WTFA_A"], bins=10
    )
    boosting_risk_deciles = _risk_decile_table(
        test["delayed_care_cost"], boosting_prob, test["WTFA_A"]
    )

    income_runs = pd.concat(
        [
            _income_model_runs(model, income, split, CORE_MODEL_COLUMNS, "core_plus_income_logistic"),
            _income_model_runs(
                model, income, split, EXPANDED_MODEL_COLUMNS, "expanded_plus_income_logistic"
            ),
        ],
        ignore_index=True,
    )
    income_summary = _summarize_income_metrics(income_runs)

    metrics.to_csv(PROCESSED_DIR / "model_metrics.csv", index=False)
    core_coef.to_csv(PROCESSED_DIR / "model_coefficients_core.csv", index=False)
    expanded_coef.to_csv(PROCESSED_DIR / "model_coefficients_expanded.csv", index=False)
    predictions.to_csv(PROCESSED_DIR / "test_predictions.csv", index=False)
    split.to_csv(PROCESSED_DIR / "model_split.csv", index=False)
    boosting_importance.to_csv(
        PROCESSED_DIR / "boosting_permutation_importance.csv", index=False
    )
    boosting_calibration.to_csv(
        PROCESSED_DIR / "boosting_calibration.csv", index=False
    )
    boosting_risk_deciles.to_csv(
        PROCESSED_DIR / "boosting_risk_deciles.csv", index=False
    )
    income_runs.to_csv(PROCESSED_DIR / "income_model_runs.csv", index=False)
    income_summary.to_csv(PROCESSED_DIR / "income_model_summary.csv", index=False)

    summary = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_positive_rows": int(test["delayed_care_cost"].sum()),
        "core_weighted_roc_auc": float(core_metrics["weighted_roc_auc"]),
        "expanded_weighted_roc_auc": float(expanded_metrics["weighted_roc_auc"]),
        "boosting_weighted_roc_auc": float(boosting_metrics["weighted_roc_auc"]),
        "core_weighted_average_precision": float(core_metrics["weighted_average_precision"]),
        "expanded_weighted_average_precision": float(expanded_metrics["weighted_average_precision"]),
        "boosting_weighted_average_precision": float(boosting_metrics["weighted_average_precision"]),
        "boosting_top_decile_positive_capture": float(
            boosting_risk_deciles.loc[
                boosting_risk_deciles["risk_decile"] == 1, "weighted_positive_share"
            ].iloc[0]
        ),
        "income_models": ["core_plus_income_logistic", "expanded_plus_income_logistic"],
        "income_imputations": 10,
        "note": "Predictive models use final survey weights as sample weights but do not use PSTRAT/PPSU for coefficient inference; population inference is reported separately in survey_analysis.py.",
    }
    (PROCESSED_DIR / "model_summary.json").write_text(json.dumps(summary, indent=2) + "
")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_modeling(), indent=2))
