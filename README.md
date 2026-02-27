# ML-Enhanced Electric Vehicle Fleet Routing Optimization

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An integrated framework combining machine learning energy prediction with mixed-integer linear programming (MILP) optimization for electric autonomous vehicle fleet routing.

## Table of Contents

1. Overview
2. Key Features
3. Results Summary
4. Project Structure
5. Installation
6. Usage
7. Data Description
8. Methodology
9. Visualizations
10. Configuration
11. Citation
12. License
13. Contact
14. References

---

## 1. Overview

Electric autonomous vehicles face operational challenges due to energy uncertainty, particularly from temperature-dependent HVAC loads that can reduce driving range by 30-40% in extreme weather. Traditional fleet routing approaches use fixed energy rates (e.g., 0.38 kWh/km) that ignore these effects, leading to suboptimal dispatch decisions.

This project develops an integrated framework that:
- Predicts trip-level energy consumption using XGBoost, capturing HVAC effects
- Optimizes vehicle-to-trip assignments using MILP with energy constraints
- Integrates ML predictions into MILP through a rolling horizon framework
- Evaluates performance across 135 scenarios with varying conditions

### Study Area

| Item | Description |
|------|-------------|
| Location | Illinois Institute of Technology campus, Chicago, IL |
| Network | 16 Points of Interest (POIs) |
| Vehicle | GreenPower EV Star electric shuttle (118 kWh battery) |
| Climate | -15°C (winter) to 38°C (summer) |

---

## 2. Key Features

| Feature | Description |
|---------|-------------|
| ML Energy Prediction | XGBoost model with R² = 0.94, temperature-aware HVAC modeling |
| MILP Optimization | Multi-objective optimization balancing energy, wait time, and service |
| Rolling Horizon Control | Real-time dispatch with configurable epoch intervals (2, 5, 10 min) |
| Comprehensive Evaluation | 135 scenarios × 3 methods = 405 simulation runs |
| Modular Architecture | Swap ML models, solvers, or network configurations |

---

## 3. Results Summary

### Overall Performance

| Metric | MILP + ML | Nearest-Available | Improvement |
|--------|-----------|-------------------|-------------|
| Service Rate | 85.4% | 73.2% | +12.2% |
| Deadhead Distance | 78 km | 124 km | -37% |
| Avg Wait Time | 142 sec | 168 sec | -15% |

### ML vs Fixed-Rate Energy Prediction

| Temperature | ML Prediction | Fixed Rate | Difference |
|-------------|---------------|------------|------------|
| Cold Winter (-15 to 0°C) | 0.44 kWh/km | 0.38 kWh/km | +17% underestimate |
| Baseline (-4 to 14°C) | 0.37 kWh/km | 0.38 kWh/km | ~0% |
| Hot Summer (28 to 38°C) | 0.32 kWh/km | 0.38 kWh/km | -18% overestimate |

### Service Rate by Fleet Size

| Fleet Size | MILP + ML | Nearest | Improvement |
|------------|-----------|---------|-------------|
| 5 vehicles | 62.3% | 45.1% | +17.2% |
| 8 vehicles | 81.7% | 68.4% | +13.3% |
| 10 vehicles | 91.2% | 82.6% | +8.6% |
| 12 vehicles | 96.8% | 93.1% | +3.7% |
| 15 vehicles | 98.5% | 97.0% | +1.5% |

---

## 4. Project Structure

```
ev-fleet-routing/
├── README.md
├── LICENSE
├── requirements.txt
├── config.yaml
│
├── data/
│   ├── raw/
│   │   ├── poi_locations.csv
│   │   └── vehicle_specs.json
│   ├── processed/
│   │   ├── trip_data.csv
│   │   ├── od_matrix.csv
│   │   └── weather_profiles.csv
│   └── results/
│       └── scenario_results.json
│
├── models/
│   └── trained_model.pkl
│
├── src/
│   ├── data_generation/
│   │   ├── generate_trips.py
│   │   ├── generate_weather.py
│   │   └── compute_energy.py
│   ├── ml/
│   │   ├── feature_engineering.py
│   │   ├── train_model.py
│   │   └── evaluate_model.py
│   ├── optimization/
│   │   ├── milp_router.py
│   │   ├── rolling_horizon.py
│   │   └── baseline_dispatch.py
│   └── analysis/
│       ├── scenario_executor.py
│       ├── analyze_results.py
│       └── visualize_results.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_ml_training.ipynb
│   ├── 03_optimization_demo.ipynb
│   └── 04_results_analysis.ipynb
│
├── tests/
│   ├── test_energy_model.py
│   ├── test_milp_solver.py
│   └── test_rolling_horizon.py
│
└── outputs/
    ├── figures/
    └── reports/
```

---

## 5. Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Setup Steps

**Step 1: Clone the repository**
```bash
git clone https://github.com/yourusername/ev-fleet-routing.git
cd ev-fleet-routing
```

**Step 2: Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
xgboost>=1.7.0
pulp>=2.7.0
highspy>=1.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
pyyaml>=6.0
tqdm>=4.65.0
```

---

## 6. Usage

### Command Line

```bash
# Generate trip data
python src/data_generation/generate_trips.py --output data/processed/trip_data.csv

# Train ML model
python src/ml/train_model.py --input data/processed/trip_data.csv --output models/trained_model.pkl

# Run simulation
python src/optimization/rolling_horizon.py \
    --trips data/processed/trip_data.csv \
    --model models/trained_model.pkl \
    --fleet-size 10 \
    --epoch-minutes 5

# Run full experiment
python src/analysis/scenario_executor.py --output data/results/scenario_results.json

# Generate figures
python src/analysis/visualize_results.py --input data/results/scenario_results.json --output outputs/figures/
```

### Python API

```python
from src.optimization.rolling_horizon import RollingHorizonController
import joblib

# Load trained model
model = joblib.load('models/trained_model.pkl')

# Initialize controller
controller = RollingHorizonController(
    fleet_size=10,
    epoch_minutes=5,
    ml_model=model,
    weights={'energy': 1.0, 'wait': 1.0, 'unserved': 1.0}
)

# Run simulation
results = controller.run(trip_data='data/processed/trip_data.csv')

# Print results
print(f"Service Rate: {results['service_rate']:.1f}%")
print(f"Total Energy: {results['total_energy']:.0f} kWh")
print(f"Avg Wait Time: {results['avg_wait_time']:.0f} sec")
```

---

## 7. Data Description

### Trip Data (trip_data.csv)

| Column | Type | Description |
|--------|------|-------------|
| trip_id | int | Unique trip identifier |
| request_time | datetime | Passenger request timestamp |
| origin_poi | str | Pickup location name |
| origin_lat | float | Pickup latitude |
| origin_lon | float | Pickup longitude |
| dest_poi | str | Dropoff location name |
| dest_lat | float | Dropoff latitude |
| dest_lon | float | Dropoff longitude |
| distance_km | float | Route distance |
| duration_sec | float | Estimated travel time |
| temperature_C | float | Ambient temperature |
| total_energy_kWh | float | Ground truth energy consumption |

### Scenario Results (scenario_results.json)

| Field | Description |
|-------|-------------|
| scenario_id | Unique scenario identifier |
| fleet_size | Number of vehicles (5, 8, 10, 12, 15) |
| epoch_minutes | Rolling horizon interval (2, 5, 10) |
| temperature_name | Weather scenario (cold_winter, baseline, hot_summer) |
| weights_name | Objective profile (energy_focused, balanced, service_focused) |
| service_rate_pct | Percentage of trips served |
| total_energy_kWh | Total energy consumed |
| avg_wait_time_sec | Average passenger wait time |
| total_deadhead_km | Non-revenue vehicle travel |

---

## 8. Methodology

### 8.1 Energy Prediction Model

XGBoost regression model with 7 input features:

| Feature | Description |
|---------|-------------|
| distance_km | Route distance (Haversine × 1.3 circuity) |
| duration_sec | Estimated travel time |
| temperature_C | Ambient temperature |
| hvac_power_kW | Temperature-dependent HVAC load |
| hour | Hour of day (0-23) |
| avg_speed_kmh | Average travel speed |
| is_peak | Peak period indicator |

**HVAC Power Function:**
```
P_HVAC(T) = 
    3.0 + 0.2 × (20 - T)    if T < 20°C   (heating)
    0.5                      if 20 ≤ T ≤ 25°C   (ventilation)
    2.0 + 0.15 × (T - 25)   if T > 25°C   (cooling)
```

**Model Performance:**

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| R² | 0.97 | 0.95 | 0.94 |
| RMSE (kWh) | 0.08 | 0.12 | 0.13 |
| MAE (kWh) | 0.05 | 0.09 | 0.10 |

### 8.2 MILP Optimization

**Objective Function:**
```
min Z = α × Σ(energy) + β × Σ(wait_time) + γ × Σ(unserved_penalty)
```

**Constraints:**
- Each trip assigned to one vehicle or marked unserved
- Vehicle availability (temporal feasibility)
- Battery capacity (SOC ≥ 20% minimum)
- Maximum wait time (≤ 10 minutes)

**Solver:** HiGHS (open-source, < 1 sec per epoch)

### 8.3 Rolling Horizon Framework

```
For each epoch (every Δ minutes):
    1. Collect pending trip requests
    2. Identify idle vehicles
    3. Predict energy for all (vehicle, trip) pairs
    4. Solve MILP for optimal assignments
    5. Execute dispatch decisions
    6. Update vehicle states (position, SOC)
```

---

## 9. Visualizations
<img width="828" height="948" alt="movement_map" src="https://github.com/user-attachments/assets/889970f6-de28-4328-926c-57ee58929dc4" />
<img width="2100" height="750" alt="energy_by_temperature" src="https://github.com/user-attachments/assets/0aa41bcd-07f3-48a8-92af-8c979547aeeb" />
<img width="1500" height="900" alt="fleet_size_sensitivity" src="https://github.com/user-attachments/assets/e195ef5a-5556-4006-bc37-66c75fdb88f0" />



---

## 10. Configuration

Key parameters in `config.yaml`:

### Vehicle Parameters
| Parameter | Value |
|-----------|-------|
| Battery capacity | 118 kWh |
| Minimum SOC | 20% |
| Motor efficiency | 85-92% |
| Regen efficiency | 60% |
| HVAC power range | 0.5-7.0 kW |

### Operational Parameters
| Parameter | Value |
|-----------|-------|
| Service hours | 7:00 AM - 10:00 PM |
| Max wait time | 10 minutes |
| Average speed | 20 km/h |
| Circuity factor | 1.3 |

### ML Parameters
| Parameter | Value |
|-----------|-------|
| Model | XGBoost |
| n_estimators | 100 |
| max_depth | 6 |
| learning_rate | 0.1 |
| Train/Val/Test split | 70/15/15 |

### Experiment Parameters
| Parameter | Values |
|-----------|--------|
| Fleet sizes | 5, 8, 10, 12, 15 |
| Epoch intervals | 2, 5, 10 minutes |
| Temperatures | cold_winter, baseline, hot_summer |
| Weight profiles | energy_focused, balanced, service_focused |

---

## 11. Citation

```bibtex
@thesis{suh2026mlfleet,
    author  = {Suh, Jungwoo},
    title   = {ML-Enhanced MILP Optimization for Electric Autonomous Vehicle Fleet Routing},
    school  = {Illinois Institute of Technology},
    year    = {2026},
    type    = {Master's Thesis},
    address = {Chicago, IL}
}
```

---

## 12. License

MIT License

Copyright (c) 2026 Jungwoo Suh

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

See LICENSE file for full text.

---

## 13. Contact

**Jungwoo Suh**  
Department of Civil, Architectural and Environmental Engineering  
Illinois Institute of Technology  
Email: jsuh10@hawk.iit.edu

### Acknowledgments

- Advisor: [Professor Name], Illinois Institute of Technology
- Vehicle specifications: GreenPower Motor Company
- HVAC modeling: Based on Kambly & Bradley (2014)
- MILP solver: HiGHS

---

## 14. References

1. Chen, T., Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.

2. Schneider, M., Stenger, A., Goeke, D. (2014). The electric vehicle-routing problem with time windows and recharging stations. *Transportation Science*, 48(4), 500-520.

3. Kambly, K.R., Bradley, T.H. (2014). Estimating the HVAC energy consumption of plug-in electric vehicles. *Journal of Power Sources*, 259, 117-124.

4. Fagnant, D.J., Kockelman, K.M. (2014). The travel and environmental implications of shared autonomous vehicles. *Sustainable Cities and Society*, 34, 127-140.

5. Huangfu, Q., Hall, J.A.J. (2018). Parallelizing the dual revised simplex method. *Mathematical Programming Computation*, 10(1), 119-142.
