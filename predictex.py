import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


DEFAULT_GEO_DATA_PATH = "world_cities_currencies.csv"
GEO_COLUMNS = {"city", "currency_code", "currency_symbol", "country"}
BUILTIN_GEO_DATA = pd.DataFrame([
    {"city": "New York", "currency_code": "USD", "currency_symbol": "$", "country": "United States"},
    {"city": "Austin", "currency_code": "USD", "currency_symbol": "$", "country": "United States"},
    {"city": "San Francisco", "currency_code": "USD", "currency_symbol": "$", "country": "United States"},
    {"city": "Chicago", "currency_code": "USD", "currency_symbol": "$", "country": "United States"},
    {"city": "Miami", "currency_code": "USD", "currency_symbol": "$", "country": "United States"},
    {"city": "Accra", "currency_code": "GHS", "currency_symbol": "GH₵", "country": "Ghana"},
])


class ExpenditurePredictor:
    def __init__(self, geo_data_path: str = DEFAULT_GEO_DATA_PATH):
        """Initializes the ML Pipeline and loads city/currency mapping."""

        data_path = Path(geo_data_path)
        if not data_path.is_absolute() and not data_path.exists():
            data_path = Path(__file__).resolve().parent / data_path
        if data_path.exists():
            self.geo_df = pd.read_csv(data_path)
        elif geo_data_path == DEFAULT_GEO_DATA_PATH:
            self.geo_df = BUILTIN_GEO_DATA.copy()
        else:
            raise FileNotFoundError(f"Currency data file not found: {geo_data_path}")
        missing_columns = GEO_COLUMNS.difference(self.geo_df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Currency data is missing required columns: {missing}")
        self.geo_df["city"] = self.geo_df["city"].fillna("").astype(str)
        self._is_trained = False

        # Define categorical and numerical features
        self.categorical_features = ['occupation', 'city']
        self.numerical_features = ['children']

        # Build preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)],
            remainder='passthrough'
        )

        # ML pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

    def get_city_currency(self, city: str) -> dict:
        """Looks up the correct local currency code and symbol for any given city."""
        if not isinstance(city, str) or not city.strip():
            raise ValueError("city must be a non-empty string")
        match = self.geo_df[self.geo_df['city'].str.casefold() == city.strip().casefold()]
        if not match.empty:
            return {
                "code": match.iloc[0]['currency_code'],
                "symbol": match.iloc[0]['currency_symbol'],
                "country": match.iloc[0]['country']
            }
        return {"code": "USD", "symbol": "$", "country": "Unknown"}

    def train(self, X: pd.DataFrame, y: np.ndarray):
        """Train the model with synthetic or real data."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        missing_features = set(self.categorical_features + self.numerical_features).difference(X.columns)
        if missing_features:
            missing = ", ".join(sorted(missing_features))
            raise ValueError(f"Training data is missing required columns: {missing}")
        y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows")
        if len(X) < 5:
            raise ValueError("At least five training rows are required")
        children = pd.to_numeric(X["children"], errors="coerce")
        if children.isna().any() or not np.isfinite(children).all():
            raise ValueError("children must contain finite numeric values")
        if np.isnan(y).any() or not np.isfinite(y).all():
            raise ValueError("y must contain finite numeric values")
        X = X.copy()
        X["children"] = children
        test_size = max(2, int(np.ceil(len(X) * 0.2)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        self.pipeline.fit(X_train, y_train)
        score = self.pipeline.score(X_test, y_test)
        self._is_trained = True
        print(f"Model successfully trained. Test R² Accuracy Score: {score:.2%}")
        return score

    def predict_local_expenditure(self, occupation: str, city: str, children: int) -> str:
        """Predicts expenditure and appends correct currency notation."""
        if not self._is_trained:
            raise RuntimeError("Train the predictor before requesting predictions")
        if not isinstance(occupation, str) or not occupation.strip():
            raise ValueError("occupation must be a non-empty string")
        if not isinstance(children, (int, np.integer)) or isinstance(children, bool) or children < 0:
            raise ValueError("children must be a non-negative integer")
        input_data = pd.DataFrame([{
            'occupation': occupation.strip(),
            'city': city.strip() if isinstance(city, str) else city,
            'children': children
        }])
        raw_prediction = self.pipeline.predict(input_data)[0]
        currency_info = self.get_city_currency(city)
        return f"{currency_info['symbol']}{raw_prediction:,.2f} {currency_info['code']} ({currency_info['country']})"


# ================================
# INTERACTIVE DEMO
# ================================
if __name__ == "__main__":
    print("--- Generating Synthetic Training Data ---")

    # Dummy occupations and cities
    occupations = ['Software Engineer', 'Teacher', 'Doctor', 'Retail Worker', 'Manager']
    cities = ['New York', 'Austin', 'San Francisco', 'Chicago', 'Miami', 'Accra']

    np.random.seed(42)
    num_samples = 1000

    # Build mock dataset
    mock_data = {
        'occupation': np.random.choice(occupations, num_samples),
        'city': np.random.choice(cities, num_samples),
        'children': np.random.randint(0, 5, num_samples)
    }
    X_df = pd.DataFrame(mock_data)

    # Synthetic spending logic
    base_costs = {'New York': 3500, 'Austin': 2200, 'San Francisco': 4500,
                  'Chicago': 2500, 'Miami': 2700, 'Accra': 1800}
    salary_weights = {'Software Engineer': 1500, 'Teacher': -500,
                      'Doctor': 3000, 'Retail Worker': -1200, 'Manager': 1000}

    y_target = np.array([
        base_costs.get(row['city'], 2500) +
        salary_weights.get(row['occupation'], 0) +
        (row['children'] * 600) +
        np.random.normal(0, 200)
        for _, row in X_df.iterrows()
    ])

    # Train model
    predictor = ExpenditurePredictor()
    predictor.train(X_df, y_target)

    # Interactive input loop
    print("\n--- Prediction Test ---")
    test_occupation = input("Enter occupation (e.g., Software Engineer, Teacher): ")
    test_city = input("Enter city (e.g., New York, Accra): ")
    try:
        test_kids = int(input("Enter number of children: "))
    except ValueError:
        raise SystemExit("Number of children must be a non-negative integer.")

    predicted_spend = predictor.predict_local_expenditure(test_occupation, test_city, test_kids)

    print("\n--- Result ---")
    print(f"Inputs: {test_occupation} in {test_city} with {test_kids} children.")
    print(f"Predicted Household Monthly Expenditure: {predicted_spend}")
