### ML Energy Prediction — Random Forest + XGBoost

Trains tree-based models to predict trip energy consumption (kWh)
from trip features. The trained model replaces the physics engine
at MILP runtime for fast energy cost lookups.

Method: Random Forest (Breiman, 2001) as primary model
        XGBoost (Chen & Guestrin, 2016) as comparison

Pipeline Position: Stage 4 (ML Energy Prediction)
    trip_data.csv → [THIS SCRIPT] → trained_model.pkl,
                                     all_models.pkl,
                                     model_evaluation.json,
                                     feature_importance.csv,
                                     test_predictions.csv

The critical design choice: the ML model never enters the MILP
formulation structure. It only affects parameter values. The physics
model generates training labels, the ML model learns to approximate
them, and the MILP uses those approximations as pre-computed
coefficients. This keeps the optimization problem linear.

References:
    - Fiori et al. (2016) - Power-based EV energy consumption model
    - Yi & Bauer (2018) - Tree-based methods for EV energy prediction
    - Breiman (2001) - Random Forests
    - Chen & Guestrin (2016) - XGBoost

Inputs:  trip_data.csv (from Fleet_simulator.py)
Outputs: trained_model.pkl      (best model + feature list for MILP)
         all_models.pkl         (both RF and XGBoost for comparison)
         model_evaluation.json  (full metrics report)
         feature_importance.csv (feature rankings for both models)
         test_predictions.csv   (actual vs predicted on test set)
