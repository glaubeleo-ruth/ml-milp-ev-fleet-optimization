import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Union

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, cross_val_score
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

try:
    from xgboost import XGBRegressor
    _XGB_AVAILABLE = True
except Exception as e:
    XGBRegressor = None  # type: ignore[misc, assignment]
    _XGB_AVAILABLE = False
    _XGB_ERROR = str(e)

import warnings
warnings.filterwarnings('ignore')


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars and arrays to native Python."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURES = [
    'distance_km',
    'n_passengers',
    'avg_speed_kmh',
    'grade_avg',
    'n_stops',
    'temperature_C',
    'hour',
]

TARGET = 'energy_kWh'

TEST_SIZE = 0.20
CV_FOLDS = 5
RANDOM_STATE = 42


# =============================================================================
# DATA LOADING & FEATURE ENGINEERING
# =============================================================================

def load_and_prepare(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load trip data and prepare features."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)

    print(f"Loaded {len(df)} trips, {len(df.columns)} columns")
    print(f"Target ({TARGET}): mean={df[TARGET].mean():.4f}, "
          f"std={df[TARGET].std():.4f}, "
          f"range=[{df[TARGET].min():.4f}, {df[TARGET].max():.4f}]")

    # Verify all features exist
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in data: {missing}")

    # Check for nulls
    nulls = df[FEATURES + [TARGET]].isnull().sum().sum()
    if nulls > 0:
        print(f"WARNING: {nulls} null values — dropping rows")
        df = df.dropna(subset=FEATURES + [TARGET])

    # Feature statistics
    print(f"\nFeature statistics:")
    for col in FEATURES:
        print(f"  {col:18s}: min={df[col].min():.3f}, "
              f"max={df[col].max():.3f}, mean={df[col].mean():.3f}")

    return df


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_random_forest(X_train, y_train):
    """
    Train Random Forest with randomized hyperparameter search.

    Random Forest: ensemble of decision trees, each trained on a
    bootstrap sample with random feature subsets. Predictions are
    averaged across all trees.

    Hyperparameter space:
        n_estimators: [100, 500] — more trees → better averaging
        max_depth: [5, 30] + None — controls tree complexity
        min_samples_split: [2, 10] — prevents overfitting on small groups
        min_samples_leaf: [1, 4] — minimum samples in leaf node
        max_features: ['sqrt', 'log2', None] — feature subsampling per split
    """
    print("\n" + "=" * 50)
    print("RANDOM FOREST — Hyperparameter Tuning")
    print("=" * 50)

    param_distributions = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 10, 15, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
    }

    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    search = RandomizedSearchCV(
        rf,
        param_distributions,
        n_iter=50,
        cv=CV_FOLDS,
        scoring='neg_mean_squared_error',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )

    search.fit(X_train, y_train)

    print(f"\nBest parameters:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
    print(f"Best CV RMSE: {np.sqrt(-search.best_score_):.6f} kWh")

    return search.best_estimator_


def train_xgboost(X_train, y_train):
    """
    Train XGBoost as comparison model.

    XGBoost builds trees sequentially — each tree corrects the errors
    of the previous ensemble. Often marginally better than RF on
    structured data, but more hyperparameters.
    """
    if not _XGB_AVAILABLE:
        raise RuntimeError(
            "XGBoost is not available. "
            "On macOS install OpenMP: brew install libomp"
        )

    print("\n" + "=" * 50)
    print("XGBOOST — Hyperparameter Tuning")
    print("=" * 50)

    param_distributions = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
    }

    xgb = XGBRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )

    search = RandomizedSearchCV(
        xgb,
        param_distributions,
        n_iter=50,
        cv=CV_FOLDS,
        scoring='neg_mean_squared_error',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )

    search.fit(X_train, y_train)

    print(f"\nBest parameters:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
    print(f"Best CV RMSE: {np.sqrt(-search.best_score_):.6f} kWh")

    return search.best_estimator_


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Evaluate model on test set."""
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # MAPE (avoid division by zero; handle all y_test <= 0)
    mask = y_test > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    else:
        mape = np.nan

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE_pct': mape,
    }

    print(f"\n--- {model_name} Test Set Performance ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    return metrics


def cross_validate(model, X, y, model_name: str) -> dict:
    """Run k-fold cross-validation."""
    scores = cross_val_score(
        model, X, y,
        cv=CV_FOLDS,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    rmse_scores = np.sqrt(-scores)

    cv_results = {
        'cv_rmse_mean': rmse_scores.mean(),
        'cv_rmse_std': rmse_scores.std(),
        'cv_rmse_scores': rmse_scores.tolist(),
    }

    print(f"\n--- {model_name} {CV_FOLDS}-Fold CV ---")
    print(f"  RMSE: {rmse_scores.mean():.6f} ± {rmse_scores.std():.6f}")
    for i, s in enumerate(rmse_scores):
        print(f"    Fold {i+1}: {s:.6f}")

    return cv_results


def feature_importance(model, feature_names: list, model_name: str) -> pd.DataFrame:
    """Extract and display feature importance."""
    importances = model.feature_importances_
    total = importances.sum()
    pct = (importances / total * 100) if total > 0 else np.zeros_like(importances)
    fi = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'importance_pct': pct
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    print(f"\n--- {model_name} Feature Importance ---")
    for _, row in fi.iterrows():
        bar = '█' * int(row['importance_pct'] / 2)
        print(f"  {row['feature']:18s} {row['importance_pct']:5.1f}%  {bar}")

    return fi


# =============================================================================
# MAIN
# =============================================================================

def main(data_dir: str = '.', output_dir: str = None):
    """
    Run the full ML training pipeline.

    Parameters
    ----------
    data_dir : str
        Directory containing trip_data.csv.
    output_dir : str
        Directory for output files. Defaults to data_dir.
    """
    print("=" * 60)
    print("ML ENERGY PREDICTION — Model Training")
    print("=" * 60)
    print(f"Features: {FEATURES}")
    print(f"Target: {TARGET}")
    print(f"Test size: {TEST_SIZE}, CV folds: {CV_FOLDS}")

    base_dir = Path(data_dir)
    out_dir = Path(output_dir) if output_dir else base_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolve trip_data.csv (search a few likely locations) ---
    candidates = [
        base_dir / "trip_data.csv",
        Path(__file__).resolve().parent / "trip_data.csv",
        Path(__file__).resolve().parent.parent / "trip_data.csv",
        Path(__file__).resolve().parent.parent / "output" / "trip_data.csv",
        Path(__file__).resolve().parent.parent / "01_Data_gen" / "Outputs" / "trip_data.csv",
    ]
    trip_data_path = None
    for p in candidates:
        if p.exists():
            trip_data_path = p
            break
    if trip_data_path is None:
        raise FileNotFoundError(
            "trip_data.csv not found. Looked in:\n  "
            + "\n  ".join(str(p) for p in candidates)
            + "\n\nGenerate it by running Fleet_simulator.py or the 01_Data_gen pipeline, "
            "then copy trip_data.csv into 02_Energy_Prediction/ or pass data_dir= to main()."
        )

    # --- Load data ---
    df = load_and_prepare(trip_data_path)

    X = df[FEATURES].values
    y = df[TARGET].values

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {len(X_train)} samples")
    print(f"Test:  {len(X_test)} samples")

    # --- Train Random Forest ---
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    rf_cv = cross_validate(rf_model, X, y, "Random Forest")
    rf_fi = feature_importance(rf_model, FEATURES, "Random Forest")

    # --- Train XGBoost (optional) ---
    if _XGB_AVAILABLE:
        xgb_model = train_xgboost(X_train, y_train)
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        xgb_cv = cross_validate(xgb_model, X, y, "XGBoost")
        xgb_fi = feature_importance(xgb_model, FEATURES, "XGBoost")
        best_model = rf_model if rf_metrics['R2'] >= xgb_metrics['R2'] else xgb_model
        best_name = "Random Forest" if rf_metrics['R2'] >= xgb_metrics['R2'] else "XGBoost"
        print(f"\n{'=' * 60}")
        print(f"MODEL COMPARISON")
        print(f"{'=' * 60}")
        print(f"{'Metric':<12} {'Random Forest':>15} {'XGBoost':>15}")
        print(f"{'-'*12} {'-'*15} {'-'*15}")
        for metric in ['RMSE', 'MAE', 'R2', 'MAPE_pct']:
            print(f"{metric:<12} {rf_metrics[metric]:>15.6f} {xgb_metrics[metric]:>15.6f}")
        print(f"\n→ Best model: {best_name}")
    else:
        xgb_model = xgb_metrics = xgb_cv = xgb_fi = None
        best_model = rf_model
        best_name = "Random Forest"
        print("\n(XGBoost not loaded — e.g. install OpenMP on macOS: brew install libomp)")
        print("→ Using Random Forest only.")

    # --- Save outputs ---
    print(f"\n{'-' * 40}")
    print("SAVING OUTPUTS")
    print(f"{'-' * 40}")

    # 1. Best model for MILP consumption
    trained_model_data = {
        'model': best_model,
        'model_name': best_name,
        'features': FEATURES,
        'target': TARGET,
    }
    with open(out_dir / 'trained_model.pkl', 'wb') as f:
        pickle.dump(trained_model_data, f)
    print(f"  trained_model.pkl — {best_name} for MILP")

    # 2. All models for comparison
    all_models_data = {
        'random_forest': {'model': rf_model, 'metrics': rf_metrics, 'cv': rf_cv},
        'best': best_name,
        'features': FEATURES,
        'target': TARGET,
    }
    if xgb_model is not None:
        all_models_data['xgboost'] = {'model': xgb_model, 'metrics': xgb_metrics, 'cv': xgb_cv}
    with open(out_dir / 'all_models.pkl', 'wb') as f:
        pickle.dump(all_models_data, f)
    print(f"  all_models.pkl — {'both models' if xgb_model is not None else 'Random Forest'} for comparison")

    # 3. Evaluation report
    evaluation = {
        'random_forest': {**rf_metrics, **rf_cv},
        'best_model': best_name,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'features': FEATURES,
    }
    if xgb_metrics is not None and xgb_cv is not None:
        evaluation['xgboost'] = {**xgb_metrics, **xgb_cv}
    with open(out_dir / 'model_evaluation.json', 'w') as f:
        json.dump(evaluation, f, indent=2, cls=_NumpyEncoder)
    print(f"  model_evaluation.json — full metrics")

    # 4. Feature importance (combined)
    rf_fi = rf_fi.copy()
    rf_fi['model'] = 'Random Forest'
    if xgb_fi is not None:
        xgb_fi = xgb_fi.copy()
        xgb_fi['model'] = 'XGBoost'
        combined_fi = pd.concat([rf_fi, xgb_fi], ignore_index=True)
    else:
        combined_fi = rf_fi
    combined_fi.to_csv(out_dir / 'feature_importance.csv', index=False)
    print(f"  feature_importance.csv — feature rankings")

    # 5. Test predictions
    y_pred_rf = rf_model.predict(X_test)
    predictions = pd.DataFrame({
        'actual': y_test,
        'predicted_rf': y_pred_rf,
        'error_rf': y_test - y_pred_rf,
    })
    if xgb_model is not None:
        y_pred_xgb = xgb_model.predict(X_test)
        predictions['predicted_xgb'] = y_pred_xgb
        predictions['error_xgb'] = y_test - y_pred_xgb
    predictions.to_csv(out_dir / 'test_predictions.csv', index=False)
    print(f"  test_predictions.csv — actual vs predicted")

    print(f"\n✓ ML training complete — {best_name} exported for MILP")

    return {
        'best_model': best_model,
        'best_name': best_name,
        'rf_metrics': rf_metrics,
        'xgb_metrics': xgb_metrics,
        'evaluation': evaluation,
    }


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "output"
    output = main(data_dir=str(script_dir), output_dir=str(output_dir))