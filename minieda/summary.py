import numpy as np
import pandas as pd

def summarize(df, include_perc=True, sort=True):
    """
    Generate a summary DataFrame with descriptive statistics for each column in the input DataFrame.

    Parameters:
        df (pd.DataFrame): The Pandas DataFrame to summarize.
        include_perc (bool, default=True): Whether to include percentage-based columns in the output.
        sort (bool, default=True): Whether to sort rows so that numeric columns appear first.

    Returns:
        pd.DataFrame: A summary table with one row per input column and one column per statistic.
    """
    numeric_cols = df.select_dtypes(include='number').columns
    non_bool_and_not_timedelta = [
        col for col in numeric_cols
        if not pd.api.types.is_bool_dtype(df[col])
        and not pd.api.types.is_timedelta64_dtype(df[col])
    ]
    
    # Use pandas describe
    desc = df.describe(include='all').T.copy()

    # Add column-level summaries
    desc['missing'] = df.isnull().sum()
    desc['missing_perc'] = (df.isnull().mean() * 100)
    desc['unique'] = df.nunique()
    desc['unique_perc'] = (desc['unique'] / len(df) * 100)
    desc['dtype'] = df.dtypes
    desc['zero'] = (df == 0).sum()
    desc['zero_perc'] = ((df == 0).mean() * 100)
    desc['skew'] = df[non_bool_and_not_timedelta].skew()

    # Round numeric summary cols to 2 decimal places
    desc = desc.assign(**{
        col: pd.to_numeric(desc[col], errors='coerce').round(2)
        for col in ['mean', 'median', 'std', 'min', '50%', 'max',
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
        desc['column_type'] = ['numeric' if pd.api.types.is_numeric_dtype(t) else 'categorical' for t in df.dtypes]
        desc = desc.sort_values(by='column_type', ascending=False)
    
    if include_perc:
        col_order = ['dtype', 'count', 'unique', 'unique_perc', 'missing', 'missing_perc', 'zero', 'zero_perc', 
                       'top', 'freq', 'mean', 'std', 'min', '50%', 'max', 'skew']
    else:
        col_order = ['dtype', 'count', 'unique', 'missing', 'zero', 
                       'top', 'freq', 'mean', 'std', 'min', '50%', 'max', 'skew']
    final_columns = [col for col in col_order if col in desc.columns]
    return desc[final_columns]