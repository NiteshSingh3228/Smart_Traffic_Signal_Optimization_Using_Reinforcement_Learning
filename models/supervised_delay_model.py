"""
Supervised Traffic Delay and Travel Time Prediction Model
Trained on validated empirical data from `data/processed/clean_tripinfo.csv` (Source 2: Archive).

Implements:
- 80/10/10 Train/Val/Test Split (preventing data leakage)
- Feature standardization
- Models evaluated: Ridge Regression, Random Forest, Gradient Boosting
- Evaluation metrics: MAE, MSE, RMSE, R² Score
- Generates delay estimates used to regularize RL rewards and evaluate trip impacts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_DIR = "results/models"
METRICS_DIR = "results/metrics"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

def train_and_evaluate_delay_models():
    print("========== TRAINING SUPERVISED TRAFFIC DELAY MODELS ==========")
    
    # 1. Load data
    df = pd.read_csv("data/processed/clean_tripinfo.csv")
    print(f"Loaded {len(df)} vehicle trip records.")
    
    # 2. Define features and targets
    feature_cols = [
        "departDelay", "departSpeed", "routeLength", "speedFactor",
        "vtype_car", "vtype_fastCar", "vtype_slowCar"
    ]
    # Add one-hot encoded approach IDs
    approach_dummies = pd.get_dummies(df["approach_id"], prefix="app", drop_first=True)
    X = pd.concat([df[feature_cols], approach_dummies], axis=1)
    
    target_col = "waitingTime"
    y = df[target_col].values
    
    print(f"Features: {list(X.columns)}")
    print(f"Target: {target_col} (mean={y.mean():.2f}, std={y.std():.2f})")
    
    # 3. Train / Validation / Test split (80% Train, 10% Val, 10% Test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1111, random_state=42  # 0.1111 * 0.9 = 0.10
    )
    
    print(f"Split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # 4. Standardize features (fitted strictly on X_train to prevent leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train candidate models
    candidates = {
        "Ridge_Regression": Ridge(alpha=1.0),
        "Random_Forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
        "Gradient_Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_val_r2 = -float("inf")
    trained_models = {}
    
    for name, model in candidates.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
        # Validation evaluation
        val_preds = model.predict(X_val_scaled)
        val_mae = mean_absolute_error(y_val, val_preds)
        val_mse = mean_squared_error(y_val, val_preds)
        val_rmse = np.sqrt(val_mse)
        val_r2 = r2_score(y_val, val_preds)
        
        # Test evaluation (unseen test set)
        test_preds = model.predict(X_test_scaled)
        test_mae = mean_absolute_error(y_test, test_preds)
        test_mse = mean_squared_error(y_test, test_preds)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, test_preds)
        
        results[name] = {
            "val_mae": round(val_mae, 3),
            "val_rmse": round(val_rmse, 3),
            "val_r2": round(val_r2, 4),
            "test_mae": round(test_mae, 3),
            "test_rmse": round(test_rmse, 3),
            "test_r2": round(test_r2, 4),
        }
        
        print(f"\n--- {name} ---")
        print(f"  Val  -> MAE: {val_mae:.2f}s, RMSE: {val_rmse:.2f}s, R²: {val_r2:.4f}")
        print(f"  Test -> MAE: {test_mae:.2f}s, RMSE: {test_rmse:.2f}s, R²: {test_r2:.4f}")
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model_name = name
            
    print(f"\nBest model selected: {best_model_name} (Val R²: {best_val_r2:.4f})")
    
    # 6. Save best model and artifacts
    best_bundle = {
        "model": trained_models[best_model_name],
        "scaler": scaler,
        "feature_names": list(X.columns),
        "target": target_col,
        "metrics": results[best_model_name]
    }
    
    model_save_path = os.path.join(MODEL_DIR, "supervised_delay_model.joblib")
    joblib.dump(best_bundle, model_save_path)
    print(f"Saved best model bundle to: {model_save_path}")
    
    metrics_save_path = os.path.join(METRICS_DIR, "supervised_delay_metrics.json")
    with open(metrics_save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_save_path}")
    
    return results

if __name__ == "__main__":
    train_and_evaluate_delay_models()

