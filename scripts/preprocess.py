import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

# Define directories
BASE_DIR = r"C:\Users\USER\Desktop\New_folder\Perovskite_stability"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# data path
DATA_PATH = os.path.join(DATA_DIR, "Perovskite.csv")

# Load data
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Original dataset size: {df.shape}")

# Filter for non-null Stability_PCE_T80
print("Filtering data...")
df = df[df['Stability_PCE_T80'].notnull()]
print(f"Filtered dataset size: {df.shape}")

# Define stability outcomes to exclude
stability_outcomes = ['Stability_PCE_end_of_experiment', 'Stability_PCE_after_1000_h']

# Identify and Remove Redundant Features
# Correlation analysis for numerical features
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = df[numerical_cols].corr()
high_corr = np.where(np.abs(corr_matrix) > 0.8)
redundant_pairs = [(corr_matrix.index[x], corr_matrix.columns[y]) 
                   for x, y in zip(*high_corr) if x != y and x < y]
features_to_drop = []
for f1, f2 in redundant_pairs:
    # Keep feature with higher variance
    if df[f1].var() > df[f2].var():
        features_to_drop.append(f2)
    else:
        features_to_drop.append(f1)

# Mutual information for categorical features
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
for col in categorical_cols:
    mi = mutual_info_regression(pd.get_dummies(df[[col]], dummy_na=True), df['Stability_PCE_T80'])
    if mi.mean() < 0.01:
        features_to_drop.append(col)

# Remove redundant features and stability outcomes
features_to_drop = list(set(features_to_drop + stability_outcomes))
df_cleaned = df.drop(columns=features_to_drop)
print(f"Features dropped: {features_to_drop}")

# Save dropped features
pd.Series(features_to_drop).to_csv(os.path.join(RESULTS_DIR, 'dropped_features.csv'), index=False)

# Feature Selection with Random Forest
X = df_cleaned.drop(columns=['Stability_PCE_T80'])
y = df_cleaned['Stability_PCE_T80']

# Impute missing values for feature importance
imputer_num = SimpleImputer(strategy='mean')
X_num = imputer_num.fit_transform(X.select_dtypes(include=['float64', 'int64']))

imputer_cat = SimpleImputer(strategy='most_frequent')
X_cat = imputer_cat.fit_transform(X.select_dtypes(include=['object', 'bool']))
X_cat_encoded = pd.get_dummies(pd.DataFrame(X_cat, columns=X.select_dtypes(include=['object', 'bool']).columns), 
                              dummy_na=True)

# Combine for Random Forest
X_imputed = np.hstack((X_num, X_cat_encoded.values))
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf.fit(X_imputed, y)
importances = pd.Series(rf.feature_importances_, 
                       index=list(X.select_dtypes(include=['float64', 'int64']).columns) + list(X_cat_encoded.columns))
top_features = importances.nlargest(20).index
X_selected = X[top_features.intersection(X.columns)]
print(f"Selected features: {X_selected.columns.tolist()}")

# Save selected features
pd.Series(X_selected.columns.tolist()).to_csv(os.path.join(RESULTS_DIR, 'selected_features.csv'), index=False)

# Preprocess Selected Features
# Impute missing values for selected features
imputer_num = SimpleImputer(strategy='mean')
X_selected_num = imputer_num.fit_transform(X_selected.select_dtypes(include=['float64', 'int64']))

cat_cols = X_selected.select_dtypes(include=['object', 'bool']).columns
if len(cat_cols) > 0:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X_selected_cat = imputer_cat.fit_transform(X_selected[cat_cols])
    # Encode categorical features
    X_selected_cat_encoded = pd.get_dummies(
        pd.DataFrame(X_selected_cat, columns=cat_cols), dummy_na=True
    )
    # Combine numerical and encoded categorical features
    X_preprocessed = np.hstack((X_selected_num, X_selected_cat_encoded.values))
    columns = list(X_selected.select_dtypes(include=['float64', 'int64']).columns) + list(X_selected_cat_encoded.columns)
else:
    X_preprocessed = X_selected_num
    columns = list(X_selected.select_dtypes(include=['float64', 'int64']).columns)

# Scale numerical features
scaler = StandardScaler()
X_preprocessed[:, :X_selected_num.shape[1]] = scaler.fit_transform(X_preprocessed[:, :X_selected_num.shape[1]])

# Create preprocessed DataFrame
preprocessed_df = pd.DataFrame(X_preprocessed, columns=columns)
preprocessed_df['Stability_PCE_T80'] = y.reset_index(drop=True)

# Save preprocessed data
preprocessed_path = os.path.join(DATA_DIR, "preprocessed_perovskite_data.csv")
preprocessed_df.to_csv(preprocessed_path, index=False)
print(f"Preprocessed data saved to {preprocessed_path}")

print("Preprocessing complete! Check 'results' and 'data' directories for outputs.")