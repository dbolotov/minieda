import numpy as np
import pandas as pd

def summarize(data, include_pct=True, sort=True):
    """
    Generate a summary DataFrame with descriptive statistics for a Pandas Series or DataFrame.

    Parameters:
        data (pd.Series or pd.DataFrame): Pandas Series or DataFrame.
        include_pct (bool, default=True): Include pctentage-based columns in the output.
        sort (bool, default=True): Sort rows so that numeric columns appear first.

    Returns:
        pd.DataFrame: A summary table with one row per input column and one column per statistic.
    """

    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError("Input must be a pandas Series or DataFrame.")

    if isinstance(data, pd.Series):
        data = data.to_frame(name=data.name or "series")

    if data.shape[1] == 0 or data.empty:
        raise ValueError("summarize() requires a non-empty Series or DataFrame with at least one column.")


    numeric_cols = data.select_dtypes(include='number').columns
    non_bool_and_not_timedelta = [
        col for col in numeric_cols
        if not pd.api.types.is_bool_dtype(data[col])
        and not pd.api.types.is_timedelta64_dtype(data[col])
    ]
    
    # Use pandas describe
    desc = data.describe(include='all').T.copy()

    # Add column-level summaries
    desc['missing'] = data.isnull().sum()
    desc['missing_pct'] = (data.isnull().mean() * 100)
    desc['unique'] = data.nunique()
    desc['unique_pct'] = (desc['unique'] / len(data) * 100)
    desc['dtype'] = data.dtypes
    desc['zero'] = (data == 0).sum()
    desc['zero_pct'] = ((data == 0).mean() * 100)
    desc['skew'] = data[non_bool_and_not_timedelta].skew()

    # Round numeric summary cols to 2 decimal places
    desc = desc.assign(**{
        col: pd.to_numeric(desc[col], errors='coerce').round(2)
        for col in ['mean', 'std', 'min', '50%', 'max',
                    'missing_pct', 'unique_pct', 'skew', 'zero_pct']
        if col in desc.columns
    })

    # Round zero pctentage to 3 decimal places
    desc = desc.assign(**{
        col: pd.to_numeric(desc[col], errors='coerce').round(3)
        for col in ['zero_pct']
        if col in desc.columns
    })

    # Ensure count is treated as integer
    desc['count'] = desc['count'].astype('Int64')

    # Replace any remaining NaNs with empty string for clean display
    desc.replace({np.nan: ""}, inplace=True)


    if sort:
        # Sort: continuous features first, categorical last
        desc['column_type'] = ['numeric' if pd.api.types.is_numeric_dtype(t) else 'categorical' for t in data.dtypes]
        desc = desc.sort_values(by='column_type', ascending=False)
    
    if include_pct:
        col_order = ['dtype', 'count', 'unique', 'unique_pct', 'missing', 'missing_pct', 'zero', 'zero_pct', 
                       'top', 'freq', 'mean', 'std', 'min', '50%', 'max', 'skew']
    else:
        col_order = ['dtype', 'count', 'unique', 'missing', 'zero', 
                       'top', 'freq', 'mean', 'std', 'min', '50%', 'max', 'skew']
    final_columns = [col for col in col_order if col in desc.columns]
    return desc[final_columns]


def summarize_ts(data, include_pct=True):
    """
    Summarize timestamp columns in a Pandas Series or DataFrame.

    Parameters:
        data (pd.Series or pd.DataFrame): A datetime Series or a DataFrame containing one or more datetime columns.
        include_pct (bool, default=True): Include pctentage-based columns in the output.

    Returns:
        pd.DataFrame: A summary table with one row per input column and one column per statistic.
    """

    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError("Input must be a pandas Series or DataFrame.")

    if data.empty:
        raise ValueError("summarize_ts() requires a non-empty Series or DataFrame.")


    if isinstance(data, pd.Series):
        if not pd.api.types.is_datetime64_any_dtype(data):
            raise ValueError("Input Series must be a datetime dtype.")
        ts_cols = [data.name or "ts"]
        ts_data = {ts_cols[0]: data}
    elif isinstance(data, pd.DataFrame):
        ts_cols = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]
        if not ts_cols:
            raise ValueError("No timestamp columns found in the input DataFrame.")
        ts_data = {col: data[col] for col in ts_cols}
    else:
        raise TypeError("Input must be a pandas Series or DataFrame.")

    n_rows = len(data)

    results = []
    for col, series in ts_data.items():
        s = series.dropna()
        is_sorted = s.is_monotonic_increasing
        row = {
            "dtype": series.dtype,
            "min": s.min(),
            "max": s.max(),
            "range": str(s.max() - s.min()) if not s.empty else "",
            "unique": s.nunique()
        }

        if include_pct:
            row["unique_pct"] = round((s.nunique() / n_rows) * 100, 2)

        row["missing"] = series.isna().sum()

        if include_pct:
            row["missing_pct"] = round((series.isna().mean()) * 100, 2)

        row["is_sorted"] = is_sorted

        results.append((col, row))

    df_result = pd.DataFrame.from_dict(dict(results), orient="index")
    
    return df_result.replace({np.nan: ""})


def summarize_missing(data):
    """
    Summarize missing values in a Pandas DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame to analyze.

    Returns:
        pd.DataFrame: A DataFrame with counts and percentages of the missing data.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if data.shape[1] == 0 or data.empty:
        raise ValueError("summarize_missing() requires a non-empty DataFrame with at least one column.")

    row_count = len(data)
    col_count = data.shape[1]

    rows_with_missing = data.isnull().any(axis=1).sum()
    cols_with_missing = data.isnull().any(axis=0).sum()
    missing_vals_total = data.isnull().sum().sum()

    rows = []

    rows.append({"metric": "rows", "count": row_count, "pct": None})
    rows.append({"metric": "cols", "count": col_count, "pct": None})

    rows.append({"metric": "rows_with_missing", "count": rows_with_missing,
                 "pct": round((rows_with_missing / row_count) * 100, 2)})

    rows.append({"metric": "cols_with_missing", "count": cols_with_missing,
                 "pct": round((cols_with_missing / col_count) * 100, 2)})

    rows.append({"metric": "missing_vals_total", "count": missing_vals_total,
                 "pct": round((missing_vals_total / (row_count * col_count)) * 100, 2)})


    df_result = pd.DataFrame(rows).set_index("metric")

    df_result = df_result.fillna("") # Replace "NaN" in the first two rows.

    return(df_result)