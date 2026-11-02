from pathlib import Path
from functools import lru_cache
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_data import (
    CORE_MODEL_COLUMNS,
    EXPANDED_MODEL_COLUMNS,
    build_adult_analysis,
    build_income_imputations,
    build_modeling_base,
    load_source_data,
    validate_source_data,
)


@lru_cache(maxsize=1)
def _prepared():
    adult, income = load_source_data()
    validate_source_data(adult, income)
    analysis = build_adult_analysis(adult)
    model = build_modeling_base(analysis)
    imps = build_income_imputations(income)
    return adult, income, analysis, model, imps


def test_source_row_counts():
    adult, income, *_ = _prepared()
    assert len(adult) == 24215
    assert len(income) == 242150


def test_one_adult_row_per_household_id():
    adult, *_ = _prepared()
    assert adult["HHX"].nunique() == 24215
    assert not adult["HHX"].duplicated().any()


def test_ten_income_imputations_per_adult():
    *_, imps = _prepared()
    counts = imps.groupby("HHX").size()
    assert (counts == 10).all()
    assert set(imps["IMPNUM_A"].unique()) == set(range(1, 11))


def test_primary_target_recode():
    *_, analysis, _, _ = _prepared()
    valid = analysis["delayed_care_cost"].dropna()
    assert set(valid.unique()) == {0, 1}
    assert int((valid == 1).sum()) == 1564
    assert int((valid == 0).sum()) == 22457
    assert int(analysis["delayed_care_cost"].isna().sum()) == 194


def test_secondary_target_is_not_in_modeling_features():
    *_, analysis, model, _ = _prepared()
    assert "forgone_care_cost" in analysis.columns
    assert "forgone_care_cost" not in model.columns
    assert "MEDNG12M_A" not in model.columns


def test_core_and_expanded_feature_sets_are_present():
    *_, model, _ = _prepared()
    assert set(CORE_MODEL_COLUMNS).issubset(model.columns)
    assert set(EXPANDED_MODEL_COLUMNS).issubset(model.columns)


def test_citizenship_is_retained_for_analysis_but_not_default_model():
    *_, analysis, model, _ = _prepared()
    assert "citizenship_status" in analysis.columns
    assert "citizenship_status" not in model.columns


def test_income_not_flattened_into_modeling_base():
    *_, model, _ = _prepared()
    assert "POVRATTC_A" not in model.columns
    assert "RATCAT_A" not in model.columns
    assert "poverty_ratio" not in model.columns


def test_survey_design_fields_are_complete():
    *_, analysis, _, _ = _prepared()
    for col in ["WTFA_A", "PSTRAT", "PPSU"]:
        assert analysis[col].notna().all()


def test_no_invalid_target_codes_survive():
    *_, analysis, _, _ = _prepared()
    assert not analysis["delayed_care_cost"].isin([7, 8, 9]).any()


def test_official_source_checksums_match_download_metadata():
    from download_data import SOURCES, md5sum

    adult_path = ROOT / "data" / "raw" / "adult25.csv"
    income_path = ROOT / "data" / "raw" / "adultinc25.csv"
    assert md5sum(adult_path) == SOURCES["adult25.csv"]["md5"]
    assert md5sum(income_path) == SOURCES["adultinc25.csv"]["md5"]
