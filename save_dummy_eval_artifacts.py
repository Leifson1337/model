import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path

# Create base directories if they don't exist
models_dir = Path("models_eval_test")
data_dir = Path("data_eval_test")
models_dir.mkdir(exist_ok=True)
data_dir.mkdir(exist_ok=True)

# 1. Dummy Model
X_train_dummy = pd.DataFrame(np.random.rand(10, 3), columns=['feat1', 'feat2', 'feat3'])
y_train_dummy = pd.Series(np.random.randint(0, 2, 10))
model = LogisticRegression()
model.fit(X_train_dummy, y_train_dummy)
model_path = models_dir / "dummy_model.joblib"
joblib.dump(model, model_path)
print(f"Dummy model saved to: {model_path}")

# 2. Dummy Scaler
scaler = MinMaxScaler()
scaler.fit(X_train_dummy) # Fit on some data
scaler_path = models_dir / "dummy_scaler.joblib"
joblib.dump(scaler, scaler_path)
print(f"Dummy scaler saved to: {scaler_path}")

# 3. Dummy Test Data
X_test_dummy_data = np.random.rand(20, 3)
# Ensure some variation for metrics calculation, especially if y_true has only one class
y_test_dummy_data_binary = np.random.randint(0, 2, 20)
if len(np.unique(y_test_dummy_data_binary)) < 2: # Ensure at least two classes for ROC AUC
    y_test_dummy_data_binary[0] = 0
    y_test_dummy_data_binary[1] = 1

test_df = pd.DataFrame(X_test_dummy_data, columns=['feat1', 'feat2', 'feat3'])
test_df['custom_target'] = y_test_dummy_data_binary # Use 'custom_target' as per eval config
test_data_path = data_dir / "dummy_test_data.csv"
test_df.to_csv(test_data_path, index=False)
print(f"Dummy test data saved to: {test_data_path}")

# 4. Dummy Evaluate Config
eval_config_content = {
    "model_path": str(model_path.resolve()), # Use absolute path for FilePath validation
    "scaler_path": str(scaler_path.resolve()),
    "test_data_path": str(test_data_path.resolve()),
    "metrics_output_json_path": "logs/dummy_eval_metrics.json",
    "metrics_to_compute": ["accuracy", "roc_auc", "f1_score", "precision", "recall", "mape", "smape"],
    "target_column": "custom_target",
    "model_type": "LogisticRegression" # For logging/potential future use
}
eval_config_path = Path("config/dummy_evaluate_config.json")
eval_config_path.parent.mkdir(exist_ok=True) # Ensure 'config' dir exists
with open(eval_config_path, 'w') as f:
    json.dump(eval_config_content, f, indent=4)
print(f"Dummy evaluate config saved to: {eval_config_path}")

print("\nDummy artifacts for evaluation test created successfully.")
