import pytest
import pandas as pd
from minieda.summary import summarize_ts


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture(scope="module")
def df_timestamps():
    return pd.DataFrame({
        "ts1": pd.date_range("2023-01-01", periods=5, freq="D"),
        "ts2": pd.date_range("2023-01-01 00:00", periods=5, freq="min"),
        "var1": [1, 2, 3, 4, 5],
        "var2": ["a", "b", "c", "d", "e"]
    })

@pytest.fixture(scope="module")
def ts_series():
    return pd.Series(pd.date_range("2023-01-01", periods=5, freq="D"), name="single_ts")

@pytest.fixture(scope="module")
def result_ts_timestamps(df_timestamps):
    return summarize_ts(df_timestamps)

@pytest.fixture(scope="module")
def result_ts_series(ts_series):
    return summarize_ts(ts_series)

# -------------------------
# Structure and Output
# -------------------------

def test_output_is_dataframe(result_ts_timestamps):
    assert isinstance(result_ts_timestamps, pd.DataFrame)

def test_index_matches_timestamp_columns(result_ts_timestamps):
    assert set(result_ts_timestamps.index) == {"ts1", "ts2"}

def test_expected_columns_present(result_ts_timestamps):
    expected = ["dtype", "min", "max", "range", "unique", "unique_pct", "missing", "missing_pct", "is_sorted"]
    for col in expected:
        assert col in result_ts_timestamps.columns

# -------------------------
# Timestamp Behavior
# -------------------------

def test_single_series_input(result_ts_series):
    assert result_ts_series.shape[0] == 1
    assert result_ts_series.index[0] == "single_ts"

def test_multiple_timestamp_columns(result_ts_timestamps):
    assert result_ts_timestamps.shape[0] == 2  # Only ts1 and ts2

def test_non_timestamp_columns_ignored(result_ts_timestamps):
    assert "var1" not in result_ts_timestamps.index
    assert "var2" not in result_ts_timestamps.index

# -------------------------
# Input Validations
# -------------------------

def test_series_with_non_timestamp_raises():
    with pytest.raises(ValueError):
        summarize_ts(pd.Series([1, 2, 3]))

def test_dataframe_with_no_timestamps_raises():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(ValueError):
        summarize_ts(df)

def test_invalid_input_type_raises():
    with pytest.raises(TypeError, match="Input must be a pandas Series or DataFrame."):
        summarize_ts(["2023-01-01", "2023-01-02"])

# -------------------------
# Value Checks
# -------------------------

def test_min_max_and_range(result_ts_timestamps, df_timestamps):
    ts1 = df_timestamps["ts1"]
    row = result_ts_timestamps.loc["ts1"]
    assert row["min"] == ts1.min()
    assert row["max"] == ts1.max()
    assert row["range"] == str(ts1.max() - ts1.min())

def test_unique_and_missing_counts(df_timestamps):
    df = df_timestamps.copy()
    df.loc[0, "ts1"] = pd.NaT
    result = summarize_ts(df)
    row = result.loc["ts1"]
    assert row["missing"] == 1
    assert row["unique"] == df["ts1"].nunique()

def test_sorted_detection():
    s_sorted = pd.Series(pd.date_range("2023-01-01", periods=5, freq="D"))
    s_unsorted = s_sorted.sample(frac=1, random_state=42)
    assert summarize_ts(s_sorted.to_frame(name="s"))['is_sorted'].iloc[0] == True
    assert summarize_ts(s_unsorted.to_frame(name="s"))['is_sorted'].iloc[0] == False

# -------------------------
# Percentage Toggle
# -------------------------

def test_include_pct_false(df_timestamps):
    result = summarize_ts(df_timestamps, include_pct=False)
    assert "missing_pct" not in result.columns
    assert "unique_pct" not in result.columns

# -------------------------
# No Side Effects
# -------------------------

def test_input_dataframe_unchanged(df_timestamps):
    original = df_timestamps.copy(deep=True)
    _ = summarize_ts(df_timestamps)
    pd.testing.assert_frame_equal(df_timestamps, original)

# -------------------------
# Series Without Name
# -------------------------

def test_series_with_no_name():
    s = pd.Series(pd.date_range("2023-01-01", periods=5))
    s.name = None
    result = summarize_ts(s)
    assert result.index[0] == "ts"

# -------------------------
# Reject Empty Series
# -------------------------
def test_empty_series_raises_error():
    s = pd.Series([], dtype='datetime64[ns]')
    with pytest.raises(ValueError, match="summarize_ts\\(\\) requires a non-empty Series or DataFrame."):
        summarize_ts(s)


# -------------------------
# Reject Empty DataFrame
# -------------------------
def test_empty_dataframe_raises_error():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="summarize_ts\\(\\) requires a non-empty Series or DataFrame."):
        summarize_ts(df)