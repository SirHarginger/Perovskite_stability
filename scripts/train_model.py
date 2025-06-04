import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Define directories
BASE_DIR = r"C:\Users\USER\Desktop\New_folder\Perovskite_stability"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Load preprocessed data
DATA_PATH = os.path.join(DATA_DIR, "preprocessed_perovskite_data.csv")
print("Loading preprocessed data...")
df = pd.read_csv(DATA_PATH)
print(f"Preprocessed dataset size: {df.shape}")

# Separate features and target
X = df.drop(columns=['Stability_PCE_T80'])
y = df['Stability_PCE_T80']

# Split data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define XGBoost parameters
print("Setting up model...")
params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'rmse'
}
model = xgb.XGBRegressor(**params)

# Train the model
print("Training model...")
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

# Perform 5-fold cross-validation
print("Performing cross-validation...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
cv_rmse = np.sqrt(-cv_scores)
print(f"Cross-Validation RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

# Visualize cross-validation scores
plt.figure(figsize=(8, 6))
sns.boxplot(x=cv_rmse)
plt.title('Cross-Validation RMSE Distribution')
plt.xlabel('RMSE')
plt.savefig(os.path.join(RESULTS_DIR, 'cv_rmse_distribution.png'))
plt.close()

# Train the model
print("Training model...")
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

# Evaluate on test set
print("Evaluating model...")
y_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {test_rmse:.4f}")

# Visualize predicted vs actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Stability_PCE_T80 (hours)')
plt.ylabel('Predicted Stability_PCE_T80 (hours)')
plt.title('Predicted vs Actual Stability_PCE_T80')
plt.savefig(os.path.join(RESULTS_DIR, 'predicted_vs_actual.png'))
plt.close()

# Visualize training and validation RMSE
evals_result = model.evals_result()
plt.figure(figsize=(8, 6))
plt.plot(evals_result['validation_0']['rmse'], label='Train RMSE')
plt.plot(evals_result['validation_1']['rmse'], label='Validation RMSE')
plt.xlabel('Boosting Rounds')
plt.ylabel('RMSE')
plt.title('Training and Validation RMSE')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'rmse_plot.png'))
plt.close()

# Feature importance
print("Calculating feature importance...")
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
importance_df = importance_df.sort_values('importance', ascending=False)
importance_df.to_csv(os.path.join(RESULTS_DIR, 'feature_importance.csv'), index=False)

# Plot feature importance
plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=importance_df.head(20))
plt.title('Top 20 Feature Importances')
plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'))
plt.close()

# SHAP analysis
print("Computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP summary plot
print("Generating SHAP summary plot...")
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig(os.path.join(RESULTS_DIR, 'shap_summary.png'))
plt.close()

# Save SHAP values
shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
shap_df.to_csv(os.path.join(RESULTS_DIR, 'shap_values.csv'), index=False)

# Save the model
print("Saving model...")
model_path = os.path.join(MODELS_DIR, 'xgb_model.json')
model.save_model(model_path)
print(f"Trained model saved to {model_path}")

print("Training complete! Check 'results' and 'models' directories for outputs.")