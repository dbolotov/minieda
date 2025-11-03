import pytest
import pandas as pd
from minieda.summary import summarize

# -------------------------
# Fixtures
# -------------------------

@pytest.fixture(scope="module")
def df_test():
    return pd.DataFrame({
        'string_col1': ['apple', 'banana', 'cherry', 'apple', 'banana', 'cherry', 'apple', 'pear', 'cherry', 'apple'],
        'string_col2': ['x', 'y', 'z', 'x', 'y', 'z', 'x', 'y', 'u', 'a'],
        'int_col1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'int_col2': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0],
        'bool_col1': [True, False, True, False, True, False, True, False, True, False],
        'bool_col2': [False, True, False, True, False, True, False, True, False, True],
        'timestamp_col1': pd.date_range("2023-01-01", periods=10, freq='D'),
        'timestamp_col2': pd.date_range("2023-06-01 00:00:00", periods=10, freq='2h'),
        'category_col': pd.Series(['low', 'medium', 'high', 'low', 'medium', 'high', 'low', 'low', 'high', 'low'], dtype='category'),
        'timedelta_col': pd.to_timedelta([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], unit='s'),
        'string_dtype_col': pd.Series(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a'], dtype='string'),
    })

@pytest.fixture(scope="module")
def result_df_test(df_test):
    return summarize(df_test)

# --------------------------------
# Structure Tests
# --------------------------------

def test_output_is_dataframe(result_df_test):
    assert isinstance(result_df_test, pd.DataFrame)

def test_rows_match_input_columns(result_df_test, df_test):
    assert result_df_test.shape[0] == df_test.shape[1]

def test_index_matches_input_columns(result_df_test, df_test):
    assert set(result_df_test.index) == set(df_test.columns)

def test_expected_columns_present(result_df_test):
    expected_cols = [
        'dtype', 'count', 'unique', 'unique_pct', 'missing', 'missing_pct',
        'zero', 'zero_pct', 'mean', 'std', 'min', '50%', 'max', 'skew'
    ]
    for col in expected_cols:
        assert col in result_df_test.columns

# --------------------------------
# Column-Specific Tests
# --------------------------------

# String columns
def test_string_column_behavior(result_df_test):
    str_col = result_df_test.loc["string_col1"]
    assert str_col["dtype"] in ("object", "string")
    assert str_col["mean"] == ""
    assert str_col["std"] == ""
    assert str_col["min"] == ""
    assert str_col["max"] == ""
    assert str_col["unique"] == 4
    assert str_col["top"] == "apple"

# Numeric column
def test_numeric_column_behavior(result_df_test):
    num_col = result_df_test.loc["int_col1"]
    assert isinstance(num_col["mean"], float)
    assert isinstance(num_col["std"], float)
    assert num_col["zero"] == 0
    assert num_col["zero_pct"] == 0.0
    assert isinstance(num_col["skew"], float)

# Boolean column
def test_boolean_column_behavior(result_df_test):
    bool_col = result_df_test.loc["bool_col1"]
    assert bool_col["dtype"] == "bool"
    assert bool_col["skew"] == ""

# Timestamp column
def test_timestamp_column_behavior(result_df_test):
    time_col = result_df_test.loc["timestamp_col1"]
    assert str(time_col["dtype"]).startswith("datetime64")
    assert time_col["mean"] == ""
    assert time_col["std"] == ""

# Category column
def test_category_column_behavior(result_df_test):
    cat_col = result_df_test.loc["category_col"]
    assert str(cat_col["dtype"]).startswith("category")
    assert cat_col["unique"] == 3

# Timedelta column
def test_timedelta_column_behavior(result_df_test):
    delta_col = result_df_test.loc["timedelta_col"]
    assert str(delta_col["dtype"]).startswith("timedelta64")
    assert delta_col["skew"] == ""

import pandas as pd
from minieda.summary import summarize

# -------------------------------
# Missing Values
# -------------------------------

def test_missing_value_summary():
    df = pd.DataFrame({
        "col1": [1, 2, None, 4, 5]
    })
    result = summarize(df)
    assert result.loc["col1", "missing"] == 1
    assert result.loc["col1", "missing_pct"] == 20.0

# -------------------------------
# Percent Control
# -------------------------------

def test_exclude_pctentage_columns(df_test):
    result = summarize(df_test, include_pct=False)
    for col in ["missing_pct", "unique_pct", "zero_pct"]:
        assert col not in result.columns

# -------------------------------
# Sorting Control
# -------------------------------

def test_no_sort_preserves_column_order(df_test):
    result = summarize(df_test, sort=False)
    assert list(result.index) == list(df_test.columns)

# -------------------------------
# Output Types
# -------------------------------

def test_numeric_output_types(result_df_test):
    numeric_cols = ["mean", "std", "min", "max", "skew", "zero_pct", "unique_pct", "missing_pct"]
    for stat in numeric_cols:
        if stat in result_df_test.columns:
            values = result_df_test[stat]
            for val in values:
                if val != "":
                    assert isinstance(val, (float, int))

def test_mean_is_rounded(result_df_test):
    if "mean" in result_df_test.columns:
        for val in result_df_test["mean"]:
            if isinstance(val, float):
                rounded = round(val, 2)
                assert abs(val - rounded) < 0.01

# -------------------------------
# No Side Effects
# -------------------------------

def test_input_dataframe_unchanged(df_test):
    df_copy = df_test.copy(deep=True)
    _ = summarize(df_test)
    pd.testing.assert_frame_equal(df_test, df_copy)

# -------------------------------
# Empty or Unusual Inputs
# -------------------------------

def test_empty_input_raises_error():
    empty_df = pd.DataFrame()
    empty_series = pd.Series(dtype=float)

    with pytest.raises(ValueError, match=r"summarize\(\) requires a non-empty Series or DataFrame with at least one column\."):
        summarize(empty_df)

    with pytest.raises(ValueError, match=r"summarize\(\) requires a non-empty Series or DataFrame with at least one column\."):
        summarize(empty_series)

def test_single_column_dataframe():
    df = pd.DataFrame({"only_col": [1, 2, 3]})
    result = summarize(df)
    assert result.shape[0] == 1
    assert "mean" in result.columns

def test_all_nan_column():
    df = pd.DataFrame({"nan_col": [None, None, None]})
    result = summarize(df)
    assert result.loc["nan_col", "missing"] == 3
    assert result.loc["nan_col", "missing_pct"] == 100.0

# -------------------------
# Test Series
# -------------------------

def test_summarize_series_numeric():
    s = pd.Series([1, 2, 3, 4], name="my_series")
    result = summarize(s)
    assert result.shape[0] == 1
    assert result.index[0] == "my_series"
    assert result.loc["my_series", "mean"] == 2.5

@pytest.mark.filterwarnings("ignore:.*Downcasting behavior in `replace` is deprecated.*")
def test_summarize_series_string():
    s = pd.Series(["a", "b", "a", "c"], name="letters")
    result = summarize(s)
    assert result.loc["letters", "unique"] == 3
    assert result.loc["letters", "top"] == "a"
