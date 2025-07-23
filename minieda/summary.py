import numpy as np
import pandas as pd

def summarize(data, include_perc=True, sort=True):
    """
    Generate a summary DataFrame with descriptive statistics for a Pandas Series or DataFrame.

    Parameters:
        data (pd.Series or pd.DataFrame): Pandas Series or DataFrame.
        include_perc (bool, default=True): Include percentage-based columns in the output.
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
    desc['missing_perc'] = (data.isnull().mean() * 100)
    desc['unique'] = data.nunique()
    desc['unique_perc'] = (desc['unique'] / len(data) * 100)
    desc['dtype'] = data.dtypes
    desc['zero'] = (data == 0).sum()
    desc['zero_perc'] = ((data == 0).mean() * 100)
    desc['skew'] = data[non_bool_and_not_timedelta].skew()

    # Round numeric summary cols to 2 decimal places
    desc = desc.assign(**{
        col: pd.to_numeric(desc[col], errors='coerce').round(2)
        for col in ['mean', 'std', 'min', '50%', 'max',
                    'missing_perc', 'unique_perc', 'skew', 'zero_perc']
        if col in desc.columns
    })

    # Round zero percentage to 3 decimal places
    desc = desc.assign(**{
        col: pd.to_numeric(desc[col], errors='coerce').round(3)
        for col in ['zero_perc']
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
    
    if include_perc:
        col_order = ['dtype', 'count', 'unique', 'unique_perc', 'missing', 'missing_perc', 'zero', 'zero_perc', 
                       'top', 'freq', 'mean', 'std', 'min', '50%', 'max', 'skew']
    else:
        col_order = ['dtype', 'count', 'unique', 'missing', 'zero', 
                       'top', 'freq', 'mean', 'std', 'min', '50%', 'max', 'skew']
    final_columns = [col for col in col_order if col in desc.columns]
    return desc[final_columns]


def summarize_ts(data, include_perc=True):
    """
    Summarize timestamp columns in a Pandas Series or DataFrame.

    Parameters:
        data (pd.Series or pd.DataFrame): A datetime Series or a DataFrame containing one or more datetime columns.
        include_perc (bool, default=True): Include percentage-based columns in the output.

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

        if include_perc:
            row["unique_perc"] = round((s.nunique() / n_rows) * 100, 2)

        row["missing"] = series.isna().sum()

        if include_perc:
            row["missing_perc"] = round((series.isna().mean()) * 100, 2)

        row["is_sorted"] = is_sorted

        results.append((col, row))

    df_result = pd.DataFrame.from_dict(dict(results), orient="index")
    return df_result.replace({np.nan: ""})
