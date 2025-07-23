# minieda

A minimalist Python package for exploratory data analysis with pandas. It currently contains two functions:

`summarize()`: an expanded version of pandas' `describe()`. Produces a table summary of a pandas Series or DataFrame, including data types, missing values, zero counts, uniqueness, distribution stats, and skew.

`summarize_ts()`: summarizes one or more datetime columns in a pandas Series or DataFrame. Ignores non-timestamp columns. Includes min/max, range, missing values, uniqueness, and whether the data is sorted.

### Why use this?

For quick insights into your data during exploratory analysis.

### Install from GitHub

```bash
pip install git+https://github.com/dbolotov/minieda.git
```

### Example - summarize
```python
import pandas as pd
from minieda import summarize

pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", None)

df = pd.DataFrame({
    "var1": [25, 30, 22, 35, 28],
    "var2": [True, False, True, True, False],
    "var3": ["A", "B", "C", "A", "B"],
    "var4": pd.date_range("2023-01-01", periods=5, freq="D"),
    "var5": pd.Series(["low", "medium", "high", "low", "medium"], dtype="category"),
})

summary = summarize(df, include_perc=True, sort=True)
print(summary)
```

Output:
```
               dtype  count  unique  unique_perc  missing  missing_perc  zero  zero_perc   top freq  mean   std   min   50%   max  skew
var1           int64      5       5        100.0        0           0.0     0        0.0             28.0  4.95  22.0  28.0  35.0  0.37
var2            bool      5       2         40.0        0           0.0     2       40.0  True    3                                    
var3          object      5       3         60.0        0           0.0     0        0.0     A    2                                    
var4  datetime64[ns]      5       5        100.0        0           0.0     0        0.0                                               
var5        category      5       3         60.0        0           0.0     0        0.0   low    2                                                                              
```

### Example: summarize_ts
```python
import pandas as pd
from minieda import summarize_ts

pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", None)

df = pd.DataFrame({
    "ts1": pd.date_range("2023-01-01", periods=5, freq="D"),
    "ts2": pd.to_datetime(["2023-01-01", "2023-01-03", None, "2023-01-05", "2020-01-04", ]),
    "val": [10, 20, 30, 40, 50],
})

summary = summarize_ts(df)
print(summary)
```

Output:
```
              dtype        min        max               range  unique  unique_perc  missing  missing_perc  is_sorted
ts1  datetime64[ns] 2023-01-01 2023-01-05     4 days 00:00:00       5        100.0        0           0.0       True
ts2  datetime64[ns] 2020-01-04 2023-01-05  1097 days 00:00:00       4         80.0        1          20.0      False
```

### Requirements
```
Python ≥ 3.8  
pandas ≥ 2.0  
numpy ≥ 1.21  
```