"""Synthetic monthly rent regression (BRL) from property and location features."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
CITIES = ["São Paulo", "Rio de Janeiro", "Belo Horizonte", "Curitiba", "Porto Alegre"]
NUMERIC = ["area_m2", "bedrooms", "bathrooms", "property_age_years", "distance_to_center_km", "neighborhood_score", "furnished"]
FEATURES = ["city"] + NUMERIC
BASE_RENT = {"São Paulo": 1100, "Rio de Janeiro": 950, "Belo Horizonte": 700, "Curitiba": 750, "Porto Alegre": 680}
RATE_PER_M2 = {"São Paulo": 43, "Rio de Janeiro": 39, "Belo Horizonte": 31, "Curitiba": 32, "Porto Alegre": 29}


def _dataset(seed=SEED, rows=2200):
    rng = np.random.default_rng(seed)
    city = rng.choice(CITIES, rows)
    X = pd.DataFrame({"city": city, "area_m2": rng.uniform(24, 220, rows),
                      "bedrooms": rng.integers(0, 6, rows), "bathrooms": rng.integers(1, 5, rows),
                      "property_age_years": rng.integers(0, 71, rows),
                      "distance_to_center_km": rng.uniform(0.5, 38, rows),
                      "neighborhood_score": rng.integers(1, 6, rows),
                      "furnished": rng.integers(0, 2, rows)})
    y = (pd.Series(city).map(BASE_RENT).to_numpy() + X.area_m2 * pd.Series(city).map(RATE_PER_M2).to_numpy()
         + 240 * X.bedrooms + 175 * X.bathrooms - 58 * X.distance_to_center_km
         + 260 * X.neighborhood_score - 8 * X.property_age_years + 430 * X.furnished
         + rng.normal(0, 420, rows))
    return X, np.maximum(500, y)


def _model(seed=SEED):
    prep = ColumnTransformer([
        ("city", OneHotEncoder(handle_unknown="ignore"), ["city"]),
        ("numeric", StandardScaler(), NUMERIC),
    ])
    return make_pipeline(prep, RandomForestRegressor(n_estimators=240, min_samples_leaf=3,
                                                     max_features=0.9, random_state=seed, n_jobs=1))


def sample_input():
    return {"city": "São Paulo", "area_m2": 78, "bedrooms": 2, "bathrooms": 2,
            "property_age_years": 12, "distance_to_center_km": 7.5, "neighborhood_score": 4, "furnished": 0}


def train_model(seed=SEED):
    X, y = _dataset(seed)
    return _model(seed).fit(X, y)


def predict(model, payload):
    row = pd.DataFrame([payload], columns=FEATURES)
    return {"estimated_monthly_rent_brl": round(max(0.0, float(model.predict(row)[0])), 2)}


def evaluate(seed=SEED):
    X, y = _dataset(seed)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    model = _model(seed).fit(x_train, y_train)
    pred = model.predict(x_test)
    return {"mae_brl": round(float(mean_absolute_error(y_test, pred)), 4),
            "rmse_brl": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
            "r2": round(float(r2_score(y_test, pred)), 4), "test_rows": int(len(y_test))}
