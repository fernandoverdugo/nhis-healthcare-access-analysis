from prepare_data import prepare_data
from survey_analysis import run_survey_analysis
from train_models import run_modeling


def main():
    prep = prepare_data()
    survey = run_survey_analysis()
    models = run_modeling()

    print("Prepared NHIS analytical datasets.")
    print(f"Adults: {prep['adult_rows']:,}")
    print(f"Valid primary outcomes: {prep['target_valid_rows']:,}")
    print(f"Delayed-care positives: {prep['target_positive_rows']:,}")
    print(f"Survey-weighted prevalence: {survey['overall_weighted_prevalence']:.2%}")
    print(f"Core logistic weighted ROC-AUC: {models['core_weighted_roc_auc']:.3f}")
    print(f"Expanded logistic weighted ROC-AUC: {models['expanded_weighted_roc_auc']:.3f}")
    print(f"Gradient boosting weighted ROC-AUC: {models['boosting_weighted_roc_auc']:.3f}")


if __name__ == "__main__":
    main()
