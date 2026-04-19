import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("backend/data/processed_travel_data.csv")

# ==============================
# REMOVE LEAKAGE
# ==============================
df = df.drop(columns=['total', 'price_hotel', 'travelCode', 'userCode', 'code'], errors='ignore')
df = df.select_dtypes(exclude=['object'])

# Remove dominant features
df = df.drop(columns=['distance', 'time'], errors='ignore')

# ==============================
# DEFINE FEATURES
# ==============================
X = df.drop('price_flight', axis=1)
y = df['price_flight']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# MLflow SETUP
# ==============================
mlflow.set_experiment("Voyage Price Prediction")

# ==============================
# MULTIPLE RUNS (IMPORTANT)
# ==============================

params_list = [
    {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1},
    {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
    {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.05}
]

for params in params_list:

    with mlflow.start_run():

        print(f"\nTraining with params: {params}")

        model = XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=0.8,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # ==============================
        # LOG PARAMETERS
        # ==============================
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.log_param("learning_rate", params["learning_rate"])

        # ==============================
        # LOG METRICS
        # ==============================
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("RMSE", rmse)

        # ==============================
        # LOG MODEL
        # ==============================
        mlflow.sklearn.log_model(model, "model")

        print(f"R2: {r2}")
        print(f"RMSE: {rmse}")

# ==============================
# SAVE FINAL MODEL
# ==============================
joblib.dump(model, "final_mlflow_model.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")

print("\nMLflow training complete!")