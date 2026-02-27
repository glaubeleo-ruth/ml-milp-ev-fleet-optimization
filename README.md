# ML-Enhanced Electric Vehicle Fleet Routing Optimization

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An integrated framework combining **Machine Learning energy prediction** with **Mixed-Integer Linear Programming (MILP) optimization** for electric autonomous vehicle fleet routing on campus networks.

![System Architecture](docs/images/system_architecture.png)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Description](#data-description)
- [Methodology](#methodology)
- [Configuration](#configuration)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

Electric autonomous vehicles face significant operational challenges due to **energy uncertainty**—particularly from temperature-dependent HVAC loads that can reduce driving range by 30-40% in extreme weather. Traditional fleet routing approaches use fixed energy rates (e.g., 0.38 kWh/km) that ignore these effects.

This project develops an integrated framework that:
1. **Predicts** trip-level energy consumption using XGBoost, capturing HVAC effects
2. **Optimizes** vehicle-to-trip assignments using MILP with energy constraints
3. **Integrates** ML predictions into MILP through a rolling horizon framework
4. **Evaluates** performance across 135 scenarios with varying temperatures, fleet sizes, and parameters

### Study Area
- **Location:** Illinois Institute of Technology (IIT) campus, Chicago, IL
- **Network:** 16 Points of Interest (POIs)
- **Vehicle:** GreenPower EV Star electric shuttle (118 kWh battery)
- **Climate:** -15°C (winter) to 38°C (summer)

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **ML Energy Prediction** | XGBoost model achieving R² = 0.94 with temperature-aware HVAC modeling |
| **MILP Optimization** | Multi-objective optimization balancing energy, wait time, and service rate |
| **Rolling Horizon Control** | Real-time dispatch with configurable epoch intervals (2, 5, 10 min) |
| **Comprehensive Evaluation** | 135 scenarios × 3 methods = 405 simulation runs |
| **Modular Architecture** | Easily swap ML models, solvers, or network configurations |

---

## 📊 Results Summary

| Metric | MILP + ML | Nearest-Available | Improvement |
|--------|-----------|-------------------|-------------|
| Service Rate | 85.4% | 73.2% | **+12.2%** |
| Deadhead Distance | 78 km | 124 km | **-37%** |
| Avg Wait Time | 142 sec | 168 sec | **-15%** |

### ML vs Fixed-Rate Energy Prediction

| Temperature | ML Prediction | Fixed Rate | Difference |
|-------------|---------------|------------|------------|
| Cold Winter (-15 to 0°C) | 0.44 kWh/km | 0.38 kWh/km | -17% error |
| Baseline (-4 to 14°C) | 0.37 kWh/km | 0.38 kWh/km | ~0% |
| Hot Summer (28 to 38°C) | 0.32 kWh/km | 0.38 kWh/km | +18% error |

---

## 📁 Project Structure

```
ev-fleet-routing/
│
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── config.yaml                     # Configuration parameters
│
├── data/
│   ├── raw/
│   │   ├── poi_locations.csv       # 16 POI coordinates and categories
│   │   └── vehicle_specs.json      # GreenPower EV Star specifications
│   ├── processed/
│   │   ├── trip_data.csv           # 4,186 generated trip requests
│   │   ├── od_matrix.csv           # Origin-destination distances
│   │   └── weather_profiles.csv    # Temperature scenarios
│   └── results/
│       └── scenario_results.json   # Experiment results (405 runs)
│
├── models/
│   └── trained_model.pkl           # Trained XGBoost energy predictor
│
├── src/
│   ├── __init__.py
│   ├── data_generation/
│   │   ├── generate_trips.py       # Trip demand generation
│   │   ├── generate_weather.py     # Temperature profile simulation
│   │   └── compute_energy.py       # Physics-based energy ground truth
│   │
│   ├── ml/
│   │   ├── feature_engineering.py  # Feature extraction
│   │   ├── train_model.py          # XGBoost training pipeline
│   │   └── evaluate_model.py       # Model evaluation metrics
│   │
│   ├── optimization/
│   │   ├── milp_router.py          # MILP formulation and solver
│   │   ├── rolling_horizon.py      # Rolling horizon controller
│   │   └── baseline_dispatch.py    # Nearest-available baseline
│   │
│   └── analysis/
│       ├── scenario_executor.py    # Full factorial experiment runner
│       ├── analyze_results.py      # Statistical analysis
│       └── visualize_results.py    # Publication-quality figures
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA and data quality checks
│   ├── 02_ml_training.ipynb        # ML model development
│   ├── 03_optimization_demo.ipynb  # MILP walkthrough
│   └── 04_results_analysis.ipynb   # Results visualization
│
├── tests/
│   ├── test_energy_model.py        # Energy calculation tests
│   ├── test_milp_solver.py         # Optimization tests
│   └── test_rolling_horizon.py     # Integration tests
│
├── docs/
│   ├── images/                     # Figures for documentation
│   └── paper/                      # Research paper drafts
│
└── outputs/
    ├── figures/                    # Generated plots
    └── reports/                    # Analysis reports
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ev-fleet-routing.git
cd ev-fleet-routing
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Requirements

```
# requirements.txt
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

## 🚀 Usage

### Quick Start

```bash
# 1. Generate trip data
python src/data_generation/generate_trips.py --output data/processed/trip_data.csv

# 2. Train ML model
python src/ml/train_model.py --input data/processed/trip_data.csv --output models/trained_model.pkl

# 3. Run single simulation
python src/optimization/rolling_horizon.py \
    --trips data/processed/trip_data.csv \
    --model models/trained_model.pkl \
    --fleet-size 10 \
    --epoch-minutes 5

# 4. Run full experiment (135 scenarios × 3 methods)
python src/analysis/scenario_executor.py --output data/results/scenario_results.json

# 5. Generate analysis and figures
python src/analysis/analyze_results.py --input data/results/scenario_results.json --output outputs/
```

### Python API

```python
from src.optimization.milp_router import FleetMILP
from src.optimization.rolling_horizon import RollingHorizonController
from src.ml.feature_engineering import build_features
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

# Print summary
print(f"Service Rate: {results['service_rate']:.1f}%")
print(f"Total Energy: {results['total_energy']:.0f} kWh")
print(f"Avg Wait Time: {results['avg_wait_time']:.0f} sec")
```

---

## 📂 Data Description

### Trip Data (`trip_data.csv`)

| Column | Type | Description |
|--------|------|-------------|
| trip_id | int | Unique trip identifier |
| request_time | datetime | When passenger requested trip |
| origin_poi | str | Pickup location (POI name) |
| origin_lat | float | Pickup latitude |
| origin_lon | float | Pickup longitude |
| dest_poi | str | Dropoff location (POI name) |
| dest_lat | float | Dropoff latitude |
| dest_lon | float | Dropoff longitude |
| distance_km | float | Route distance (Haversine × 1.3) |
| duration_sec | float | Estimated travel time |
| temperature_C | float | Ambient temperature at trip time |
| total_energy_kWh | float | Ground truth energy (physics model) |

### Scenario Results (`scenario_results.json`)

```json
{
  "metadata": {
    "total_scenarios": 135,
    "total_runs": 405,
    "timestamp": "2026-02-18T10:30:00"
  },
  "results": [
    {
      "config": {
        "scenario_id": 1,
        "fleet_size": 5,
        "epoch_minutes": 2,
        "temperature_name": "cold_winter",
        "weights_name": "balanced"
      },
      "methods": {
        "milp_ml": {
          "trips_served": 2612,
          "trips_missed": 1574,
          "service_rate_pct": 62.4,
          "total_energy_kWh": 287.3,
          "avg_wait_time_sec": 156.2,
          "total_deadhead_km": 89.4
        },
        "milp_fixed": { ... },
        "nearest": { ... }
      }
    }
  ]
}
```

---

## 🔬 Methodology

### 1. Energy Prediction (XGBoost)

**Features:**
- Distance (km)
- Duration (sec)
- Temperature (°C)
- HVAC power (kW)
- Hour of day
- Peak indicator

**HVAC Power Function:**
```
P_HVAC(T) = 
  3.0 + 0.2×(20-T)     if T < 20°C  (heating)
  0.5                   if 20 ≤ T ≤ 25°C  (ventilation)
  2.0 + 0.15×(T-25)    if T > 25°C  (cooling)
```

### 2. MILP Optimization

**Objective:**
```
min Z = α·Σ(energy) + β·Σ(wait_time) + γ·Σ(unserved_penalty)
```

**Constraints:**
- Trip assignment (each trip → one vehicle or unserved)
- Vehicle availability (temporal feasibility)
- Battery capacity (SOC ≥ minimum)
- Maximum wait time (≤ 10 minutes)

### 3. Rolling Horizon

```
For each epoch (every Δ minutes):
  1. Collect pending trip requests
  2. Identify idle vehicles
  3. Predict energy for all (vehicle, trip) pairs
  4. Solve MILP for optimal assignments
  5. Execute dispatch decisions
  6. Update vehicle states
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize parameters:

```yaml
# config.yaml

network:
  poi_file: "data/raw/poi_locations.csv"
  circuity_factor: 1.3
  avg_speed_kmh: 20

vehicle:
  battery_capacity_kwh: 118
  min_soc_pct: 20
  hvac_power_min_kw: 0.5
  hvac_power_max_kw: 7.0

operation:
  service_start: "07:00"
  service_end: "22:00"
  max_wait_minutes: 10
  boarding_time_sec: 30

ml:
  model_type: "xgboost"
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15

optimization:
  solver: "highs"
  time_limit_sec: 60
  mip_gap: 0.01

experiment:
  fleet_sizes: [5, 8, 10, 12, 15]
  epoch_minutes: [2, 5, 10]
  temperatures: ["cold_winter", "baseline", "hot_summer"]
  weight_profiles: ["energy_focused", "balanced", "service_focused"]
```

---

## 📖 Citation

If you use this code in your research, please cite:

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Jungwoo Suh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 📧 Contact

**Jungwoo Suh**  
Department of Civil, Architectural and Environmental Engineering  
Illinois Institute of Technology  
Email: jsuh10@hawk.iit.edu

---

## 🙏 Acknowledgments

- Advisor: [Professor Name], Illinois Institute of Technology
- Vehicle data: [GreenPower Motor Company](https://greenpowermotor.com/)
- HVAC modeling: Based on Kambly & Bradley (2014)
- MILP solver: [HiGHS](https://highs.dev/)

---

## 📊 Visualizations

All figures are generated from `scenario_results.json` using publication-quality settings (300 DPI, serif fonts).

### Figure 1: Service Rate by Fleet Size
![Service Rate by Fleet Size](outputs/figures/fig1_service_rate_by_fleet.png)

Line chart comparing service rates across three dispatch methods as fleet size increases from 5 to 15 vehicles.

**Key Finding:** MILP outperforms Nearest-Available by +17.2% with 5 vehicles, decreasing to +1.5% with 15 vehicles.

---

### Figure 2: MILP Improvement Over Nearest
![MILP Improvement](outputs/figures/fig2_milp_improvement.png)

Bar chart showing percentage point improvement from MILP optimization.

**Key Finding:** Optimization provides greatest value for constrained fleets (5-8 vehicles).

---

### Figure 3: Energy Consumption by Temperature
![Energy by Temperature](outputs/figures/fig3_energy_by_temperature.png)

Grouped bar chart comparing energy consumption across temperature scenarios.

**Key Finding:** Cold winter has highest consumption (~387 kWh) due to HVAC heating loads.

---

### Figure 4: ML vs Fixed Rate Energy Difference
![ML vs Fixed Energy](outputs/figures/fig4_ml_vs_fixed_energy.png)

Bar chart showing prediction difference between ML and fixed-rate (0.38 kWh/km).

**Key Finding:** 
- Cold Winter: +17% (fixed rate underestimates)
- Hot Summer: -18% (fixed rate overestimates)

---

### Figure 5: Epoch Interval Sensitivity
![Epoch Sensitivity](outputs/figures/fig5_epoch_sensitivity.png)

Dual-axis chart showing trade-off between epoch length, service rate, and wait time.

**Key Finding:** 5-minute epochs balance optimization (+0.8% service) with wait time (+17 sec).

---

### Figure 6: Performance Distribution Box Plots
![Box Plots](outputs/figures/fig6_boxplot_comparison.png)

Three-panel box plot showing distribution of service rate, energy, and wait time.

**Key Finding:** MILP methods have higher median and tighter distribution than Nearest.

---

### Figure 7: Service Rate Heatmap
![Heatmap](outputs/figures/fig7_heatmap_service_rate.png)

Heatmaps showing service rate by fleet size × temperature for each method.

**Key Finding:** MILP+ML achieves >90% service with 10+ vehicles in all conditions.

---

### Figure 8: Objective Weight Sensitivity
![Weight Sensitivity](outputs/figures/fig8_weight_sensitivity.png)

Grouped bar chart showing how objective weights affect performance.

**Key Finding:** Service-focused weights: +3.9% service rate at cost of +10% energy.

---

### Figure 9: Feature Importance
![Feature Importance](outputs/figures/fig9_feature_importance.png)

Horizontal bar chart showing XGBoost feature importance.

**Key Finding:** Distance (42%) + HVAC-related features (38%) dominate predictions.

---

### Figure 10: Deadhead Comparison
![Deadhead Comparison](outputs/figures/fig10_deadhead_comparison.png)

Bar chart comparing non-revenue driving distance.

**Key Finding:** MILP reduces deadhead by **37%** (124 km → 78 km).

---

### Generating Figures

```bash
python src/analysis/visualize_results.py \
    --input data/results/scenario_results.json \
    --output outputs/figures/
```

### Color Scheme

| Method | Color | Hex |
|--------|-------|-----|
| MILP + ML | Green | `#2ecc71` |
| MILP + Fixed | Blue | `#3498db` |
| Nearest-Available | Red | `#e74c3c` |

---

## 📚 References

Key references used in this project:

1. Chen, T., Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
2. Schneider, M., et al. (2014). The electric vehicle-routing problem with time windows. *Transportation Science*.
3. Kambly, K.R., Bradley, T.H. (2014). Estimating the HVAC energy consumption of plug-in EVs. *Journal of Power Sources*.
4. Fagnant, D.J., Kockelman, K.M. (2014). Shared autonomous vehicles. *Sustainable Cities and Society*.

See full reference list in the [research paper](docs/paper/Research_Paper_Final.pdf).
