import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 1 & 2. Load the cleaned and raw datasets
df_clean = pd.read_csv('fleet_data_cleaned.csv')
df_raw = pd.read_csv('raw_fleet_data.csv')

X = df_clean.drop(columns=['Target_CO2_Emissions'])
y = df_clean['Target_CO2_Emissions']

# 3. Create rigorous Train (70%), Validation (15%), and Hold-out Test (15%) splits
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)

# 4. Define stochastic hyperparameters for robustness
xgb_params = {
    'n_estimators': 1000,          
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,              
    'colsample_bytree': 0.8,       
    'tree_method': 'hist',
    'device': 'cuda',
    'early_stopping_rounds': 50  
}

# 5. Train main model with Early Stopping and GPU fallback
try:
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
except Exception as e:
    print(f"CUDA unavailable or fallback triggered... (Error: {e})")
    xgb_params['device'] = 'cpu'
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

# 6. Rigorous Repeated Cross-Validation (stripping early stopping for sklearn compatibility)
cv_params = xgb_params.copy()
cv_params.pop('early_stopping_rounds', None)
cv_model = xgb.XGBRegressor(**cv_params)

rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
cv_scores = cross_val_score(cv_model, X_train, y_train, cv=rkf, scoring='r2')

# 7. Evaluate on the untouched Hold-out Test Set
y_pred_test = model.predict(X_test)
r2 = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)

print("--- Professional Model Diagnostics ---")
print(f"Optimal Trees Found:  {model.best_iteration}")
print(f"Repeated 5-Fold R²:   {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
print(f"Hold-out Test R²:     {r2:.4f}")
print(f"Hold-out Test RMSE:   {rmse:.4f}")
print(f"Hold-out Test MAE:    {mae:.4f}\n")

print("--- Feature Importances ---")
feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
print(feat_imp_df.sort_values(by='Importance', ascending=False).to_string(index=False))
print("-" * 35)

# 8. Prevent Data Leakage in Thresholding
high_threshold = np.percentile(y_train, 85)
medium_threshold = np.percentile(y_train, 50)

def categorize_priority(val):
    if val >= high_threshold:
        return 'High Priority'
    elif val >= medium_threshold:
        return 'Medium Priority'
    else:
        return 'Normal'

# 9. Generate final predictions and export
all_predictions = model.predict(X)

df_export = df_raw.copy()
df_export['Predicted_CO2_Emissions'] = np.round(all_predictions, 2)
df_export['Absolute_Error'] = np.round(np.abs(df_export['Target_CO2_Emissions'] - df_export['Predicted_CO2_Emissions']), 2)
df_export['Replacement_Priority'] = df_export['Predicted_CO2_Emissions'].apply(categorize_priority)

df_export.to_csv('model_predictions.csv', index=False)
print("\n[✓] Export complete. Leakage-free predictions saved to 'model_predictions.csv'.")
model.save_model("xgb_model.json")
print("\n[✓] Model successfully saved to 'xgb_model.json'")