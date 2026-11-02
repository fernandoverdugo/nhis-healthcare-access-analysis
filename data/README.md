# Data architecture

This project uses the **2025 National Health Interview Survey (NHIS) Sample Adult public-use file** and the separate **Sample Adult multiply imputed income file**.

## Raw inputs

The pipeline expects the following local files in `data/raw/`:

- `adult25.csv`
- `adultinc25.csv`

The raw source files are excluded from Git history. Run `python scripts/download_data.py` from the repository root to download the official NCHS archives and verify the published MD5 checksums.

Respondent-level derived intermediates are also excluded from Git history. They are regenerated locally from the official source files; only aggregate/model-summary outputs needed for review are committed.

## Processed preparation outputs

### `adult_analysis.csv` *(local intermediate; not committed)*

One row per Sample Adult. Contains the selected source variables, cleaned labels, primary outcome (`delayed_care_cost`), secondary outcome (`forgone_care_cost`), curated predictors, and survey-design fields.

### `modeling_base.csv` *(local intermediate; not committed)*

One row per Sample Adult with the default core and expanded predictive feature sets. It intentionally excludes:

- `forgone_care_cost` / `MEDNG12M_A` to avoid target leakage
- citizenship status from the default predictive specification
- income variables, which are handled through the separate multiply imputed file

### `income_imputations.csv` *(local intermediate; not committed)*

Long-format table containing the 10 official poverty-ratio imputations for every Sample Adult. Each `HHX` therefore appears exactly 10 times, once for `IMPNUM_A` 1–10.

### `variable_dictionary.csv`

Project-specific dictionary documenting selected source variables, analytical role, and treatment.

### `data_summary.json`

Validation counts and basic outcome prevalence from the preparation stage.

## Survey-analysis outputs

### `survey_overall.csv`

Overall survey-weighted delayed-care prevalence, Taylor-linearized standard error, 95% confidence interval, and design degrees of freedom.

### `survey_group_estimates.csv`

Survey-domain estimates by insurance status, usual place of care, collapsed education, region, general health, functioning difficulty, nativity, urban-rural classification, and housing tenure.

### `income_pooled_estimates.csv`

Survey-weighted delayed-care prevalence by four family-poverty bands. Estimates are produced separately for all 10 completed income datasets and then pooled so both survey-design variance and between-imputation uncertainty contribute to the final standard error. The output also records the fraction of missing information and Barnard–Rubin adjusted multiple-imputation degrees of freedom.

### `survey_summary.json`

Compact metadata and headline survey result.

## Predictive-modeling outputs

### `model_split.csv` *(local intermediate; not committed)*

Fixed train/test assignment by `HHX`, using a stratified 75/25 split with random state 2026.

### `model_metrics.csv`

Held-out metrics for the prevalence baseline, core logistic regression, expanded logistic regression, and expanded histogram gradient-boosting benchmark.

### `test_predictions.csv` *(local intermediate; not committed)*

Held-out probabilities for the baseline, core logistic, expanded logistic, and gradient-boosting models. This supports reproducible ROC, precision-recall, calibration, and risk-ranking diagnostics.

### `model_coefficients_core.csv` and `model_coefficients_expanded.csv`

Regularized logistic coefficients and exponentiated coefficients. They are descriptive model diagnostics, not survey-adjusted inferential odds ratios.

### `boosting_permutation_importance.csv`

Held-out permutation importance for the original expanded-model features, scored with survey-weighted ROC-AUC. These values describe predictive reliance, not causal effects.

### `boosting_calibration.csv`

Ten equally sized held-out risk bins with weighted mean predicted probability and weighted observed delayed-care prevalence.

### `boosting_risk_deciles.csv`

Highest-to-lowest predicted-risk deciles with weighted observed prevalence, positive-outcome share, and cumulative positive-outcome capture.

### `income_model_runs.csv`

Performance from the core-plus-income and expanded-plus-income models fit separately for imputations 1–10.

### `income_model_summary.csv`

Mean, standard deviation, minimum, and maximum performance across the 10 income imputations.

### `model_subgroup_metrics.csv`

Held-out subgroup diagnostics for sex, race/ethnicity, and nativity for the expanded logistic and gradient-boosting models. The file reports subgroup sample sizes, weighted prevalence, mean predicted probability, calibration gap, ROC-AUC, average precision, and Brier score. Small subgroups are retained transparently and should be interpreted cautiously rather than treated as fairness conclusions.

### `model_summary.json`

Compact modeling metadata and headline discrimination metrics.

## Primary outcome

`MEDDL12M_A` asks whether the Sample Adult delayed medical care during the past 12 months because of cost. Valid responses are recoded to 1/0; refused, not ascertained, and don't know responses become missing.

`MEDNG12M_A` is a related cost-related forgone-care outcome. It is retained for secondary analysis but is never used to predict `MEDDL12M_A`.

## Survey design

Population estimates use:

- `WTFA_A` — final annual Sample Adult weight
- `PSTRAT` — annual public variance stratum
- `PPSU` — annual public variance PSU

The 2025 public-use design represented in this project has 46 public strata and 608 stratum-PSU combinations, yielding 562 design degrees of freedom in the primary analysis.

## Income treatment

The ordinary Sample Adult file contains poverty-ratio values based on the first imputation only. The separate income file contains 10 plausible values per Sample Adult. This project preserves that structure.

For population estimates, each completed income dataset is analyzed separately and the resulting point estimates and variances are pooled. For predictive analysis, the same model specification is refit separately for each imputation and performance variation is reported explicitly.

## Source attribution and license scope

**Data Source:** National Center for Health Statistics, National Health Interview Survey, 2025.

See `../DATA_USE.md` for the NCHS data-use conditions, requested attribution/disclaimer, and the distinction between the repository's MIT-licensed original code and the NHIS source data.
