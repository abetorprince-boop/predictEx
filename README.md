# PredictEx

**PredictEx** is a small Python library for predicting local household monthly expenditure based on occupation, city, and number of children and automatically formatting the result in the correct local currency for that city.

It wraps a scikit-learn `RandomForestRegressor` inside a preprocessing pipeline (one-hot encoding for categorical fields) and pairs it with a city → currency lookup table, so predictions come back as a ready-to-display string like `GH₵2,150.00 GHS (Ghana)` instead of a raw number.

---

## Features

- 🔧 **Ready-made ML pipeline**: one-hot encodes `occupation` and `city`, passes `children` through, and feeds a `RandomForestRegressor`.
- 🌍 **City → currency resolution**: looks up the correct currency code, symbol, and country for a given city, with a graceful `USD ($)` fallback for unknown cities.
- 📄 **Configurable geo data** — load your own city/currency CSV, or fall back to a small built-in table if none is supplied.
- ✅ **Input validation** — clear errors for malformed training data, missing columns, non-numeric fields, and untrained-model usage.
- 🎮 **Interactive CLI demo** — run the file directly to train on synthetic data and try live predictions.

---

## Installation

PredictEx isn't packaged for PyPI — drop `predictex.py` into your project and install its dependencies:

```bash
pip install numpy pandas scikit-learn
```

**Requirements:**
- Python 3.8+
- `numpy`
- `pandas`
- `scikit-learn`

---

## Quick Start

```python
import numpy as np
import pandas as pd
from predictex import ExpenditurePredictor

# 1. Prepare training data
X = pd.DataFrame({
    "occupation": ["Software Engineer", "Teacher", "Doctor", "Retail Worker", "Manager"],
    "city":       ["Accra", "Accra", "New York", "Miami", "Chicago"],
    "children":   [0, 2, 1, 3, 0],
})
y = np.array([1800, 2600, 5200, 2100, 2700])

# 2. Train
predictor = ExpenditurePredictor()
predictor.train(X, y)

# 3. Predict
result = predictor.predict_local_expenditure(
    occupation="Software Engineer",
    city="Accra",
    children=1,
)
print(result)  # e.g. "GH₵2,015.32 GHS (Ghana)"
```

You can also run the module directly for an interactive demo trained on synthetic data:

```bash
python predictex.py
```

This generates 1,000 synthetic samples, trains the model, and then prompts you for an occupation, city, and number of children before printing a predicted expenditure.

---


### `get_city_currency(city: str) -> dict`

Looks up currency info for a city, case-insensitively.

- Returns `{"code": ..., "symbol": ..., "country": ...}`.
- Unknown cities fall back to `{"code": "USD", "symbol": "$", "country": "Unknown"}`.
- Raises `ValueError` if `city` is empty or not a string.

### `train(X: pd.DataFrame, y: np.ndarray) -> float`

Trains the pipeline on the given data and returns the R² score on a held-out test split.

Requirements:
- `X` must be a `pandas.DataFrame` containing `occupation`, `city`, and `children` columns.
- At least **5 rows** of training data are required.
- `children` must be finite and numeric (coerced with `pd.to_numeric`).
- `y` must be finite and numeric, and the same length as `X`.
- Internally uses an 80/20 train/test split (minimum test size of 2 rows) with `random_state=42`.

Raises `TypeError` / `ValueError` on malformed input.

### `predict_local_expenditure(occupation: str, city: str, children: int) -> str`

Predicts expenditure for a single case and returns it as a formatted currency string.

- Requires the model to have been trained first (`RuntimeError` otherwise).
- `occupation` must be a non-empty string.
- `children` must be a non-negative integer (booleans are rejected).
- The returned string has the form: `{symbol}{amount:,.2f} {currency_code} ({country})`.

---


---

## Notes & Limitations

- The model is a `RandomForestRegressor` with fixed hyperparameters (`n_estimators=100`, `random_state=42`)  no built-in hyperparameter tuning.
- Categorical features (`occupation`, `city`) use `OneHotEncoder(handle_unknown="ignore")`, so unseen categories at prediction time won't error, but also won't contribute meaningfully to the prediction.
- Currency formatting is purely a lookup step; it does not perform live currency conversion.
- The `__main__` block is a demo only it trains on **synthetic, randomly generated** data and should not be used as a real expenditure model.

---

## License

 MIT
