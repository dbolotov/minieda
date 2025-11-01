import pytest
import pandas as pd
from minieda.summary import summarize_missing


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture(scope="module")
def df_missing():
    return pd.DataFrame({
        "col1": [1, None, 3, None, 5],
        "col2": [None, 2, 3, 4, 5],
        "col3": [1, 2, 3, 4, 5],
        "col4": [None, None, None, None, None],
    })

@pytest.fixture(scope="module")
def df_no_missing():
    return pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    })

@pytest.fixture(scope="module")
def result_missing(df_missing):
    return summarize_missing(df_missing)

@pytest.fixture(scope="module")
def result_no_missing(df_no_missing):
    return summarize_missing(df_no_missing)

# -------------------------
# Structure and Output
# -------------------------

def test_output_is_dataframe(result_missing):
    assert isinstance(result_missing, pd.DataFrame)
    assert result_missing.shape[0] == 5
    assert result_missing.shape[1] == 2
    assert result_missing.index.is_unique

def test_expected_rows_present(result_missing):
    expected = ["rows","cols","rows_with_missing","cols_with_missing","missing_vals_total"]
    assert list(result_missing.index) == expected

# -------------------------
# Correctness Checks
# -------------------------

def test_missing_counts_pct_correct(result_missing):
    assert result_missing.loc["rows", "count"] == 5
    assert result_missing.loc["cols", "count"] == 4
    assert result_missing.loc["rows_with_missing", "count"] == 5
    assert result_missing.loc["cols_with_missing", "count"] == 3
    assert result_missing.loc["missing_vals_total", "count"] == 8

    assert result_missing.loc["rows_with_missing", "pct"] == 100.0
    assert result_missing.loc["cols_with_missing", "pct"] == 75.0
    assert result_missing.loc["missing_vals_total", "pct"] == 40.0

def test_pct_column_blank_when_not_applicable(result_missing):
    assert result_missing.loc["rows", "pct"] == ""
    assert result_missing.loc["cols", "pct"] == ""

def test_no_missing_returns_zeroes(result_no_missing):
    assert result_no_missing.loc["rows_with_missing", "count"] == 0
    assert result_no_missing.loc["cols_with_missing", "count"] == 0
    assert result_no_missing.loc["missing_vals_total", "count"] == 0

    assert result_no_missing.loc["rows_with_missing", "pct"] == 0.0
    assert result_no_missing.loc["cols_with_missing", "pct"] == 0.0
    assert result_no_missing.loc["missing_vals_total", "pct"] == 0.0

# -------------------------
# Input Validations
# -------------------------

def test_invalid_input_raises():
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame."):
        summarize_missing([1,2,3])

def test_empty_dataframe_raises():
    with pytest.raises(ValueError):
        summarize_missing(pd.DataFrame())

# -------------------------
# No Side Effects
# -------------------------

def test_input_dataframe_unchanged(df_missing):
    original = df_missing.copy(deep=True)
    summarize_missing(df_missing)
    pd.testing.assert_frame_equal(df_missing, original)