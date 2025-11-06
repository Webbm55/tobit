# tobit
Simple Tobit regression in Python

Tobit regression is a form of censored regression that can handle a mix of left- and right-censored (and, of course, uncensored) observations of the target variable.

The `R` package `censReg` guided much of the implementation.

## Dependencies

- numpy
- pandas
- scipy
- scikit-learn

## Usage

```python
from tobit import TobitModel
import pandas as pd
import numpy as np

# Create TobitModel instance
model = TobitModel(fit_intercept=True)

# Fit the model
# x: DataFrame of predictor variables
# y: Series of target values
# cens: Series of censoring indicators (-1: left-censored, 0: uncensored, 1: right-censored)
model.fit(x, y, cens)

# Make predictions
predictions = model.predict(x_test)
```

See `tobit.ipynb` for complete examples including:
- Artificial censored regression data
- Comparison with R censReg package using AER Affairs dataset