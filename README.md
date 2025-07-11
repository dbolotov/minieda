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

df = pd.DataFrame({
    "var1": [25, 30, 22, 35, 28],
    "var2": [50000, 60000, 52000, 75000, 58000],
    "var3": ["A", "B", "C", "A", "B"]
})

print(summarize(df))
```

```
       dtype  count  unique  unique_perc  missing  missing_perc  zero  zero_perc top freq     mean      std      min      50%      max  skew
var1   int64      5       5        100.0        0           0.0     0        0.0              28.0     4.95     22.0     28.0     35.0  0.37
var2   int64      5       5        100.0        0           0.0     0        0.0           59000.0  9848.86  50000.0  58000.0  75000.0  1.32
var3  object      5       3         60.0        0           0.0     0        0.0   A    2                                                   

```

## Requirements

- Python ≥ 3.8  
- pandas ≥ 2.0  
- numpy ≥ 1.21  
