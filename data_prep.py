"""
Municipal Fleet Emissions - end-to-end pipeline
===============================================
1. Generates the synthetic fleet dataset (generator unchanged, same seed -> same data).
2. Splits FIRST (70 / 15 / 15), then fits target encoding + scaling on the training rows only.
3. Trains XGBoost with early stopping on the validation set, evaluates on the hold-out test set.
4. Exports everything the dashboard needs.

Run:  python fleet_pipeline.py
Then copy xgb_model.json, preprocessing.json and model_metadata.json next to app.py.
The three files come from the same run and must always be replaced together.

Outputs
-------
raw_fleet_data.csv        raw synthetic dataset
fleet_data_cleaned.csv    encoded + scaled features (train-fitted) and target
model_predictions.csv     predictions for every row, labelled by data split
xgb_model.json            trained model
preprocessing.json        encodings, scaler values and the data ranges the model was trained on
model_metadata.json       evaluation metrics, parameters, versions
"""

import datetime as dt
import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

SEED = 42
N = 5_000
TARGET = 'Target_CO2_Emissions'
CAT_COLS = ['Vehicle_Class', 'Fuel_Type', 'Route_Type']
RAW_FEATURES = CAT_COLS + ['Vehicle_Age_Years', 'Cumulative_Mileage']
FEATURE_COLS = ['Vehicle_Age_Years', 'Cumulative_Mileage',
                'Vehicle_Class_encoded', 'Fuel_Type_encoded', 'Route_Type_encoded']

# ===========================================================================
# PART 1 - DATA GENERATION (unchanged from the original script)
# ===========================================================================
np.random.seed(SEED)

vehicle_ids = [f"MUN_{i:04d}" for i in range(1, N + 1)]

vehicle_classes = np.random.choice(
    ["Transit Bus", "Waste Truck", "Service Van"], size=N, p=[0.40, 0.30, 0.30])

fuel_types = []
for vc in vehicle_classes:
    if vc == "Service Van":
        fuel_types.append(np.random.choice(["Petrol", "Diesel"], p=[0.50, 0.50]))
    else:  # Transit Bus or Waste Truck
        fuel_types.append(np.random.choice(["Diesel", "CNG", "Hybrid"], p=[0.60, 0.25, 0.15]))
fuel_types = np.array(fuel_types)

route_types = np.random.choice(
    ["Urban Stop-and-Go", "Mixed Suburban", "Highway Transit"], size=N, p=[0.50, 0.30, 0.20])

vehicle_age = np.random.randint(1, 16, size=N)                 # 1 - 15 years
annual_mileage = np.random.randint(15_000, 40_001, size=N)
cumulative_mileage = vehicle_age * annual_mileage

baseline_map = {"Service Van": 200.0, "Transit Bus": 1000.0, "Waste Truck": 1300.0}
baseline = np.array([baseline_map[vc] for vc in vehicle_classes], dtype=float)

fuel_modifier = np.ones(N, dtype=float)
fuel_modifier[fuel_types == "CNG"] *= 0.85
fuel_modifier[fuel_types == "Hybrid"] *= 0.75

route_modifier = np.ones(N, dtype=float)
route_modifier[route_types == "Urban Stop-and-Go"] *= 1.25
route_modifier[route_types == "Highway Transit"] *= 0.90

age_penalty = 1.0 + (vehicle_age * 0.015)
mileage_penalty = 1.0 + ((cumulative_mileage / 100_000) * 0.02)

co2_signal = baseline * fuel_modifier * route_modifier * age_penalty * mileage_penalty
noise = np.random.normal(0, 15, size=N)
co2_emissions = co2_signal + noise

raw_df = pd.DataFrame({
    "Vehicle_ID": vehicle_ids,
    "Vehicle_Class": vehicle_classes,
    "Fuel_Type": fuel_types,
    "Route_Type": route_types,
    "Vehicle_Age_Years": vehicle_age,
    "Cumulative_Mileage": cumulative_mileage,
    "Target_CO2_Emissions": co2_emissions,
})
raw_df.to_csv("raw_fleet_data.csv", index=False)
print(f"[✓] raw_fleet_data.csv saved -> shape: {raw_df.shape}")

# ===========================================================================
# PART 2 - SPLIT FIRST, THEN FIT PREPROCESSING ON TRAINING ROWS ONLY
# ===========================================================================
# Same seeds and sizes as before, so the row partition is unchanged.
train_val_raw, test_raw = train_test_split(raw_df, test_size=0.15, random_state=SEED)
train_raw, val_raw = train_test_split(train_val_raw, test_size=0.1765, random_state=SEED)

split_label = pd.Series('train', index=raw_df.index)
split_label.loc[val_raw.index] = 'validation'
split_label.loc[test_raw.index] = 'test'


class FleetPreprocessor(BaseEstimator, TransformerMixin):
    """Target-encodes the categorical columns, then z-scores all five model features.
    Everything is learned in fit(), so fitting on training rows keeps validation/test rows out."""

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(np.asarray(y, dtype=float))
        self.global_mean_ = float(y.mean())
        self.maps_ = {c: y.groupby(X[c]).mean().to_dict() for c in CAT_COLS}
        self.scaler_ = StandardScaler().fit(self._encode(X))
        return self

    def _encode(self, X):
        out = pd.DataFrame({
            'Vehicle_Age_Years': X['Vehicle_Age_Years'].to_numpy(dtype=float),
            'Cumulative_Mileage': X['Cumulative_Mileage'].to_numpy(dtype=float),
        })
        for c in CAT_COLS:
            out[f'{c}_encoded'] = X[c].map(self.maps_[c]).fillna(self.global_mean_).to_numpy(dtype=float)
        return out[FEATURE_COLS]

    def transform(self, X):
        X = pd.DataFrame(X)
        scaled = self.scaler_.transform(self._encode(X.reset_index(drop=True)))
        return pd.DataFrame(scaled, columns=FEATURE_COLS, index=X.index)


prep = FleetPreprocessor().fit(train_raw[RAW_FEATURES], train_raw[TARGET])
X_train, y_train = prep.transform(train_raw[RAW_FEATURES]), train_raw[TARGET]
X_val, y_val = prep.transform(val_raw[RAW_FEATURES]), val_raw[TARGET]
X_test, y_test = prep.transform(test_raw[RAW_FEATURES]), test_raw[TARGET]
print(f"[✓] Preprocessing fitted on {len(train_raw):,} training rows only "
      f"(validation {len(val_raw):,}, test {len(test_raw):,}).")
print(f"    Encodings: { {c: {k: round(v, 2) for k, v in m.items()} for c, m in prep.maps_.items()} }")

# ===========================================================================
# PART 3 - TRAINING
# ===========================================================================
xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'early_stopping_rounds': 50,
    'random_state': SEED,
}


def fit_model(device: str):
    params = {**xgb_params, 'device': device}
    m = xgb.XGBRegressor(**params)
    # Early stopping watches the LAST eval_set entry, i.e. the validation set.
    m.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    return m, params


try:
    model, used_params = fit_model('cuda')
except xgb.core.XGBoostError as e:
    print(f"CUDA unavailable, falling back to CPU... (Error: {e})")
    model, used_params = fit_model('cpu')

best_n_trees = int(model.best_iteration) + 1      # best_iteration is zero-based

# Leak-free repeated CV: the preprocessor is re-fitted inside every fold.
cv_params = {k: v for k, v in used_params.items() if k != 'early_stopping_rounds'}
cv_params['n_estimators'] = best_n_trees
cv_pipe = Pipeline([('prep', FleetPreprocessor()), ('xgb', xgb.XGBRegressor(**cv_params))])
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
cv_scores = cross_val_score(cv_pipe, train_raw[RAW_FEATURES], train_raw[TARGET], cv=rkf, scoring='r2')


def metrics(X_, y_):
    p = model.predict(X_)
    return {'r2': float(r2_score(y_, p)),
            'rmse': float(np.sqrt(mean_squared_error(y_, p))),
            'mae': float(mean_absolute_error(y_, p))}


m_train, m_val, m_test = metrics(X_train, y_train), metrics(X_val, y_val), metrics(X_test, y_test)

print("\n--- Model Diagnostics ---")
print(f"Trees used:           {best_n_trees}")
print(f"Repeated 5-Fold R²:   {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
print(f"Train R²:             {m_train['r2']:.4f}")
print(f"Validation R²:        {m_val['r2']:.4f}")
print(f"Hold-out Test R²:     {m_test['r2']:.4f}")
print(f"Hold-out Test RMSE:   {m_test['rmse']:.4f}")
print(f"Hold-out Test MAE:    {m_test['mae']:.4f}")
print("(The data is synthetic and generated from a formula plus N(0, 15) noise, so a near-perfect R² "
      "is expected and says nothing about accuracy on a real fleet.)\n")

print("--- Feature Importances ---")
feat_imp_df = pd.DataFrame({'Feature': FEATURE_COLS, 'Importance': model.feature_importances_})
print(feat_imp_df.sort_values(by='Importance', ascending=False).to_string(index=False))

# ===========================================================================
# PART 4 - EXPORTS
# ===========================================================================
high_threshold = np.percentile(y_train, 85)
medium_threshold = np.percentile(y_train, 50)


def categorize_priority(val):
    if val >= high_threshold:
        return 'High Priority'
    elif val >= medium_threshold:
        return 'Medium Priority'
    return 'Normal'


X_all = prep.transform(raw_df[RAW_FEATURES])
pd.concat([X_all, raw_df[TARGET]], axis=1).to_csv("fleet_data_cleaned.csv", index=False)

df_export = raw_df.copy()
df_export['Data_Split'] = split_label.to_numpy()
df_export['Predicted_CO2_Emissions'] = np.round(model.predict(X_all), 2)
df_export['Absolute_Error'] = np.round(np.abs(df_export[TARGET] - df_export['Predicted_CO2_Emissions']), 2)
df_export['Replacement_Priority'] = df_export['Predicted_CO2_Emissions'].apply(categorize_priority)
df_export.to_csv("model_predictions.csv", index=False)
print("\n[✓] model_predictions.csv saved (use Data_Split == 'test' rows for honest error figures).")

model.save_model("xgb_model.json")

# --- preprocessing.json: exactly what the dashboard needs to reproduce the features ---
keys = ['age', 'mileage', 'class', 'fuel', 'route']
km_per_year = train_raw['Cumulative_Mileage'] / train_raw['Vehicle_Age_Years']
fuels_by_class = {c: sorted(g['Fuel_Type'].unique().tolist()) for c, g in train_raw.groupby('Vehicle_Class')}
preprocessing = {
    'created_at': dt.datetime.now().isoformat(timespec='seconds'),
    'fit_on': 'training split only',
    'feature_cols': FEATURE_COLS,
    'means': dict(zip(keys, [float(v) for v in prep.scaler_.mean_])),
    'stds': dict(zip(keys, [float(v) for v in prep.scaler_.scale_])),
    'class_map': {k: float(v) for k, v in prep.maps_['Vehicle_Class'].items()},
    'fuel_map': {k: float(v) for k, v in prep.maps_['Fuel_Type'].items()},
    'route_map': {k: float(v) for k, v in prep.maps_['Route_Type'].items()},
    'domain': {
        'age': [int(train_raw['Vehicle_Age_Years'].min()), int(train_raw['Vehicle_Age_Years'].max())],
        'km_per_year': [float(km_per_year.min()), float(km_per_year.max())],
        'fuels_by_class': fuels_by_class,
    },
}
with open('preprocessing.json', 'w') as f:
    json.dump(preprocessing, f, indent=2)

metadata = {
    'trained_at': dt.datetime.now().isoformat(timespec='seconds'),
    'xgboost_version': xgb.__version__,
    'features': FEATURE_COLS,
    'data': {'source': 'synthetic', 'rows_total': N, 'seed': SEED},
    'rows': {'train': len(train_raw), 'validation': len(val_raw), 'test': len(test_raw)},
    'best_n_trees': best_n_trees,
    'params': dict(used_params),
    'cv': {'scheme': 'RepeatedKFold 5x3, preprocessing re-fitted per fold',
           'r2_mean': float(np.mean(cv_scores)), 'r2_std': float(np.std(cv_scores))},
    'train': m_train, 'validation': m_val, 'test': m_test,
    'percentile_thresholds': {'medium_p50': float(medium_threshold), 'high_p85': float(high_threshold)},
}
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("[✓] Saved xgb_model.json, preprocessing.json and model_metadata.json - copy all three next to app.py.")
print("\nClass x Fuel combinations in the training data:")
print(pd.crosstab(train_raw['Vehicle_Class'], train_raw['Fuel_Type']).to_string())