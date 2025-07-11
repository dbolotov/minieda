# minieda

A minimalist utility for exploratory data analysis in Python using pandas DataFrames.

`summarize()`: One-line summary of any DataFrame.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/dbolotov/minieda.git
```

## Usage

```python
import pandas as pd
from minieda import summarize

df = pd.read_csv("your_dataset.csv")
summary = summarize(df)
print(summary)
```

## Example Output
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
```

```
               dtype  count  unique  unique_perc  missing  missing_perc  zero  zero_perc   top freq  mean   std   min   50%   max  skew
var1           int64      5       5        100.0        0           0.0     0        0.0             28.0  4.95  22.0  28.0  35.0  0.37
var2            bool      5       2         40.0        0           0.0     2       40.0  True    3                                    
var3          object      5       3         60.0        0           0.0     0        0.0     A    2                                    
var4  datetime64[ns]      5       5        100.0        0           0.0     0        0.0                                               
var5        category      5       3         60.0        0           0.0     0        0.0   low    2                                                                               

```

## Requirements

- Python ≥ 3.8  
- pandas ≥ 2.0  
- numpy ≥ 1.21  
