import numpy as np
import pandas as pd
import json
import time as _time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from itertools import product
import sys

# Import rolling horizon controller (lives in 03_MILP)
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent / "03_MILP"))
from rolling_horizon import RollingHorizonController


# =============================================================================
# SCENARIO CONFIGURATION
# =============================================================================

@dataclass
class ObjectiveWeights:
    """MILP objective function weights."""
    name: str
    alpha: float   # energy weight
    beta: float    # wait time weight
    gamma: float   # unserved penalty


@dataclass
class TemperatureProfile:
    """Temperature modification for trip data."""
    name: str
    temp_min: float      # None = keep original
    temp_max: float      # None = keep original
    description: str


# Pre-defined weight profiles
WEIGHT_PROFILES = {
    'balanced': ObjectiveWeights('balanced', alpha=1.0, beta=0.01, gamma=10.0),
    'energy_focused': ObjectiveWeights('energy_focused', alpha=2.0, beta=0.005, gamma=5.0),
    'service_focused': ObjectiveWeights('service_focused', alpha=0.5, beta=0.02, gamma=20.0),
}

# Temperature scenarios
TEMP_PROFILES = {
    'cold_winter': TemperatureProfile(
        'cold_winter', -15.0, 0.0,
        'Chicago winter: heavy heating HVAC load'
    ),
    'baseline': TemperatureProfile(
        'baseline', None, None,
        'Original data: mild temps (-3.6 to 13.5°C)'
    ),
    'hot_summer': TemperatureProfile(
        'hot_summer', 28.0, 38.0,
        'Chicago summer: heavy cooling HVAC load'
    ),
}

# Fleet size options
FLEET_SIZES = [5, 8, 10, 12, 15]

# Epoch interval options (minutes)
EPOCH_INTERVALS = [2.0, 5.0, 10.0]


@dataclass
class ScenarioConfig:
    """Complete scenario specification."""
    scenario_id: str
    fleet_size: int
    epoch_minutes: float
    weights: ObjectiveWeights
    temperature: TemperatureProfile

    def to_dict(self) -> dict:
        return {
            'scenario_id': self.scenario_id,
            'fleet_size': self.fleet_size,
            'epoch_minutes': self.epoch_minutes,
            'weights_name': self.weights.name,
            'alpha': self.weights.alpha,
            'beta': self.weights.beta,
            'gamma': self.weights.gamma,
            'temperature_name': self.temperature.name,
            'temperature_desc': self.temperature.description,
        }


# =============================================================================
# TEMPERATURE MODIFICATION
# =============================================================================

def modify_temperature(
    trip_data: pd.DataFrame,
    temp_profile: TemperatureProfile,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Modify trip temperatures to match a target profile.

    For 'baseline', returns original data unchanged.
    For other profiles, replaces temperature_C with values sampled
    from a distribution matching diurnal patterns.
    """
    df = trip_data.copy()

    if temp_profile.name == 'baseline' or temp_profile.temp_min is None:
        return df

    np.random.seed(seed)
    n = len(df)

    # Create diurnal temperature pattern
    hours = df['hour'].values
    diurnal_offset = 0.5 * np.sin((hours - 6) * np.pi / 12)

    # Base temperature (uniform within range)
    t_min, t_max = temp_profile.temp_min, temp_profile.temp_max
    t_range = t_max - t_min
    base_temps = np.random.uniform(t_min + 0.3 * t_range, t_max - 0.3 * t_range, n)

    # Apply diurnal variation (±3°C swing)
    new_temps = base_temps + diurnal_offset * 3.0

    # Clip to profile bounds
    new_temps = np.clip(new_temps, t_min, t_max)

    df['temperature_C'] = new_temps.round(1)
    return df


# =============================================================================
# SCENARIO EXECUTOR
# =============================================================================

class ScenarioExecutor:
    """
    Execute scenarios and collect results.

    Each scenario runs 3 methods:
        1. MILP + ML Energy Prediction
        2. MILP + Fixed Energy Rate
        3. Nearest-Available Baseline
    """

    def __init__(
        self,
        trip_data: pd.DataFrame,
        station_info: List[Dict],
        vehicle_starts_template: List[Dict],
        ml_model_path: str,
        output_dir: Path,
    ):
        self.trip_data_base = trip_data.copy()
        self.station_info = station_info
        self.vehicle_starts_template = vehicle_starts_template
        self.ml_model_path = ml_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = []
        self.failed_scenarios = []

    def _get_vehicle_starts(self, n_vehicles: int) -> List[Dict]:
        """Get vehicle starting positions for given fleet size."""
        if n_vehicles <= len(self.vehicle_starts_template):
            return self.vehicle_starts_template[:n_vehicles]
        else:
            extended = []
            for i in range(n_vehicles):
                base = self.vehicle_starts_template[
                    i % len(self.vehicle_starts_template)
                ].copy()
                base['id'] = i
                extended.append(base)
            return extended

    def run_scenario(
        self,
        config: ScenarioConfig,
        verbose: bool = False,
    ) -> Dict:
        """
        Execute a single scenario with all three methods.
        Returns results for all methods plus deltas.
        """
        print(f"\n  Scenario: {config.scenario_id}")
        print(f"    Fleet={config.fleet_size}, Epoch={config.epoch_minutes}min, "
              f"Weights={config.weights.name}, Temp={config.temperature.name}")

        # Modify temperature
        trip_data = modify_temperature(self.trip_data_base, config.temperature)
        vehicle_starts = self._get_vehicle_starts(config.fleet_size)

        scenario_results = {
            'config': config.to_dict(),
            'methods': {},
            'deltas': {},
        }

        # --- Method 1: MILP + ML ---
        print("    [1/3] MILP + ML...")
        t0 = _time.time()
        try:
            rh_ml = RollingHorizonController(
                trip_data=trip_data,
                station_info=self.station_info,
                vehicle_starts=vehicle_starts,
                ml_model_path=self.ml_model_path,
                epoch_minutes=config.epoch_minutes,
                n_vehicles=config.fleet_size,
                alpha=config.weights.alpha,
                beta=config.weights.beta,
                gamma=config.weights.gamma,
            )
            results_ml = rh_ml.run(strategy='milp', verbose=False)
            results_ml['method'] = 'milp_ml'
            results_ml['wall_time_total'] = round(_time.time() - t0, 2)
            scenario_results['methods']['milp_ml'] = results_ml
            print(f"      Service rate: {results_ml['service_rate_pct']}%")
        except Exception as e:
            print(f"      ERROR: {e}")
            scenario_results['methods']['milp_ml'] = {'error': str(e)}

        # --- Method 2: MILP + Fixed Rate ---
        print("    [2/3] MILP + Fixed Rate...")
        t0 = _time.time()
        try:
            rh_fixed = RollingHorizonController(
                trip_data=trip_data,
                station_info=self.station_info,
                vehicle_starts=vehicle_starts,
                ml_model_path=None,  # No ML → fixed 0.38 kWh/km
                epoch_minutes=config.epoch_minutes,
                n_vehicles=config.fleet_size,
                alpha=config.weights.alpha,
                beta=config.weights.beta,
                gamma=config.weights.gamma,
            )
            results_fixed = rh_fixed.run(strategy='milp', verbose=False)
            results_fixed['method'] = 'milp_fixed'
            results_fixed['wall_time_total'] = round(_time.time() - t0, 2)
            scenario_results['methods']['milp_fixed'] = results_fixed
            print(f"      Service rate: {results_fixed['service_rate_pct']}%")
        except Exception as e:
            print(f"      ERROR: {e}")
            scenario_results['methods']['milp_fixed'] = {'error': str(e)}

        # --- Method 3: Nearest-Available ---
        print("    [3/3] Nearest-Available...")
        t0 = _time.time()
        try:
            rh_greedy = RollingHorizonController(
                trip_data=trip_data,
                station_info=self.station_info,
                vehicle_starts=vehicle_starts,
                ml_model_path=None,
                epoch_minutes=config.epoch_minutes,
                n_vehicles=config.fleet_size,
            )
            results_baseline = rh_greedy.run(strategy='greedy', verbose=False)
            results_baseline['method'] = 'nearest'
            results_baseline['wall_time_total'] = round(_time.time() - t0, 2)
            scenario_results['methods']['nearest'] = results_baseline
            print(f"      Service rate: {results_baseline['service_rate_pct']}%")
        except Exception as e:
            print(f"      ERROR: {e}")
            scenario_results['methods']['nearest'] = {'error': str(e)}

        # Compute deltas
        scenario_results['deltas'] = self._compute_deltas(scenario_results['methods'])

        self.results.append(scenario_results)
        return scenario_results

    def _compute_deltas(self, methods: Dict) -> Dict:
        """Compute improvement metrics between methods."""
        deltas = {}

        ml = methods.get('milp_ml', {})
        fixed = methods.get('milp_fixed', {})
        nearest = methods.get('nearest', {})

        # ML vs Fixed (isolates ML contribution)
        if 'error' not in ml and 'error' not in fixed:
            fixed_energy = fixed.get('total_energy_kWh', 1)
            ml_energy = ml.get('total_energy_kWh', 1)
            energy_pct = (
                (fixed_energy - ml_energy) / fixed_energy * 100
                if fixed_energy > 0 else 0
            )

            deltas['ml_vs_fixed'] = {
                'trips_diff': ml.get('trips_served', 0) - fixed.get('trips_served', 0),
                'energy_diff_kWh': round(fixed_energy - ml_energy, 2),
                'energy_pct_reduction': round(energy_pct, 2),
                'wait_diff_sec': round(
                    fixed.get('avg_wait_sec', 0) - ml.get('avg_wait_sec', 0), 2
                ),
            }

        # MILP (ML) vs Nearest (overall optimization benefit)
        if 'error' not in ml and 'error' not in nearest:
            deltas['milp_vs_nearest'] = {
                'trips_diff': ml.get('trips_served', 0) - nearest.get('trips_served', 0),
                'service_rate_diff': round(
                    ml.get('service_rate_pct', 0) - nearest.get('service_rate_pct', 0), 2
                ),
                'energy_diff_kWh': round(
                    nearest.get('total_energy_kWh', 0) - ml.get('total_energy_kWh', 0), 2
                ),
                'deadhead_diff_km': round(
                    nearest.get('total_deadhead_km', 0) - ml.get('total_deadhead_km', 0), 2
                ),
            }

        return deltas

    # -----------------------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------------------

    def save_results(self, filename: str = 'scenario_results.json'):
        """Save full results as JSON."""
        path = self.output_dir / filename
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to {path}")

    def save_summary_csv(self, filename: str = 'scenario_summary.csv') -> pd.DataFrame:
        """Save flattened summary to CSV for easy analysis."""
        rows = []
        for res in self.results:
            cfg = res['config']
            for method_name, method_res in res['methods'].items():
                if 'error' in method_res:
                    continue
                row = {
                    'scenario_id': cfg['scenario_id'],
                    'fleet_size': cfg['fleet_size'],
                    'epoch_minutes': cfg['epoch_minutes'],
                    'weights': cfg['weights_name'],
                    'temperature': cfg['temperature_name'],
                    'method': method_name,
                    'trips_served': method_res.get('trips_served'),
                    'trips_missed': method_res.get('trips_unserved'),
                    'service_rate_pct': method_res.get('service_rate_pct'),
                    'total_energy_kWh': method_res.get('total_energy_kWh'),
                    'avg_energy_per_trip': method_res.get('avg_energy_per_trip'),
                    'avg_wait_sec': method_res.get('avg_wait_sec'),
                    'total_deadhead_km': method_res.get('total_deadhead_km'),
                    'wall_time_sec': method_res.get('wall_time_total',
                                                     method_res.get('wall_time_sec')),
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        print(f"Summary saved to {path}")
        return df


# =============================================================================
# SCENARIO GENERATION
# =============================================================================

def generate_all_scenarios() -> List[ScenarioConfig]:
    """Generate full factorial of all scenario dimensions."""
    scenarios = []
    for fleet, epoch, wname, tname in product(
        FLEET_SIZES, EPOCH_INTERVALS,
        WEIGHT_PROFILES.keys(), TEMP_PROFILES.keys()
    ):
        sid = f"F{fleet}_E{int(epoch)}_W{wname[:3]}_T{tname[:3]}"
        scenarios.append(ScenarioConfig(
            scenario_id=sid,
            fleet_size=fleet,
            epoch_minutes=epoch,
            weights=WEIGHT_PROFILES[wname],
            temperature=TEMP_PROFILES[tname],
        ))
    return scenarios


def generate_sweep_scenarios(sweep_type: str) -> List[ScenarioConfig]:
    """
    Generate scenarios for a single-dimension sweep.
    Holds other dimensions at baseline values.
    """
    scenarios = []
    baseline_w = WEIGHT_PROFILES['balanced']
    baseline_t = TEMP_PROFILES['baseline']

    if sweep_type == 'fleet':
        for fleet in FLEET_SIZES:
            sid = f"sweep_fleet_F{fleet}"
            scenarios.append(ScenarioConfig(
                scenario_id=sid,
                fleet_size=fleet,
                epoch_minutes=5.0,
                weights=baseline_w,
                temperature=baseline_t,
            ))
    elif sweep_type == 'epoch':
        for epoch in EPOCH_INTERVALS:
            sid = f"sweep_epoch_E{int(epoch)}"
            scenarios.append(ScenarioConfig(
                scenario_id=sid,
                fleet_size=10,
                epoch_minutes=epoch,
                weights=baseline_w,
                temperature=baseline_t,
            ))
    elif sweep_type == 'temperature':
        for tname, tprofile in TEMP_PROFILES.items():
            sid = f"sweep_temp_{tname}"
            scenarios.append(ScenarioConfig(
                scenario_id=sid,
                fleet_size=10,
                epoch_minutes=5.0,
                weights=baseline_w,
                temperature=tprofile,
            ))
    elif sweep_type == 'weights':
        for wname, wprofile in WEIGHT_PROFILES.items():
            sid = f"sweep_weight_{wname}"
            scenarios.append(ScenarioConfig(
                scenario_id=sid,
                fleet_size=10,
                epoch_minutes=5.0,
                weights=wprofile,
                temperature=baseline_t,
            ))
    else:
        raise ValueError(f"Unknown sweep type: {sweep_type}")

    return scenarios


def generate_key_scenarios() -> List[ScenarioConfig]:
    """
    Generate a curated set of key scenarios that capture the most
    important variations without running the full factorial.
    """
    baseline_w = WEIGHT_PROFILES['balanced']
    baseline_t = TEMP_PROFILES['baseline']

    scenarios = [
        # Baseline
        ScenarioConfig('key_baseline', 10, 5.0, baseline_w, baseline_t),
        # Small fleet (resource-constrained)
        ScenarioConfig('key_small_fleet', 5, 5.0, baseline_w, baseline_t),
        # Large fleet (resource-abundant)
        ScenarioConfig('key_large_fleet', 15, 5.0, baseline_w, baseline_t),
        # Cold winter (HVAC heating)
        ScenarioConfig('key_cold', 10, 5.0, baseline_w, TEMP_PROFILES['cold_winter']),
        # Hot summer (HVAC cooling)
        ScenarioConfig('key_hot', 10, 5.0, baseline_w, TEMP_PROFILES['hot_summer']),
        # Energy-focused objective
        ScenarioConfig('key_energy', 10, 5.0, WEIGHT_PROFILES['energy_focused'], baseline_t),
    ]
    return scenarios


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='EV Shuttle Scenario Executor'
    )
    parser.add_argument('--all', action='store_true',
                        help='Run full factorial (135 scenarios)')
    parser.add_argument('--key', action='store_true',
                        help='Run key scenarios (6)')
    parser.add_argument('--sweep', type=str, default=None,
                        choices=['fleet', 'epoch', 'temperature', 'weights'],
                        help='Run single-dimension sweep')
    parser.add_argument('--fleet', type=int, default=10)
    parser.add_argument('--epoch', type=float, default=5.0)
    parser.add_argument('--weights', type=str, default='balanced',
                        choices=WEIGHT_PROFILES.keys())
    parser.add_argument('--temp', type=str, default='baseline',
                        choices=TEMP_PROFILES.keys())
    parser.add_argument('--data-dir', type=str, default='.')
    parser.add_argument('--output-dir', type=str, default=None)

    args = parser.parse_args()

    base_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / 'scenario_outputs'

    # Resolve trip_data.csv (search likely locations)
    _script_dir = Path(__file__).resolve().parent
    _project_root = _script_dir.parent
    _trip_candidates = [
        base_dir / 'trip_data.csv',
        base_dir / '01_Data_gen' / 'Outputs' / 'trip_data.csv',
        _project_root / '01_Data_gen' / 'Outputs' / 'trip_data.csv',
        _script_dir / 'trip_data.csv',
    ]
    _trip_path = next((p for p in _trip_candidates if p.exists()), None)
    if _trip_path is None:
        raise FileNotFoundError(
            "trip_data.csv not found. Tried:\n  " + "\n  ".join(str(p) for p in _trip_candidates)
            + "\nUse --data-dir or place trip_data.csv in 01_Data_gen/Outputs/."
        )
    trip_data = pd.read_csv(_trip_path)
    print(f"Loaded {len(trip_data)} trips from {_trip_path}")

    model_path = str(base_dir / 'trained_model.pkl')

    # Station info (6 selected stations)
    station_info = [
        {'name': 'MTCC', 'lat': 41.838385, 'lon': -87.627555, 'chargers': 2},
        {'name': 'Paul Galvin Library', 'lat': 41.833675, 'lon': -87.628336, 'chargers': 2},
        {'name': 'McCormick Student Village', 'lat': 41.835527, 'lon': -87.624207, 'chargers': 2},
        {'name': 'Crown Hall', 'lat': 41.833199, 'lon': -87.627273, 'chargers': 2},
        {'name': 'Kaplan Institute', 'lat': 41.836861, 'lon': -87.628300, 'chargers': 2},
        {'name': 'Arthur S. Keating Sports Center', 'lat': 41.838985, 'lon': -87.625566, 'chargers': 2},
    ]

    # Vehicle starting positions (template — trimmed per fleet size)
    vehicle_starts_template = [
        {'id': 0, 'lat': 41.837866, 'lon': -87.624703},  # Kacek Hall
        {'id': 1, 'lat': 41.831394, 'lon': -87.627231},  # Michael Galvin Tower
        {'id': 2, 'lat': 41.833199, 'lon': -87.627273},  # S.R. Crown Hall
        {'id': 3, 'lat': 41.835681, 'lon': -87.628387},  # Herman Hall
        {'id': 4, 'lat': 41.836861, 'lon': -87.628300},  # Kaplan Institute
        {'id': 5, 'lat': 41.835527, 'lon': -87.624207},  # McCormick Student Village
        {'id': 6, 'lat': 41.833675, 'lon': -87.628336},  # Paul Galvin Library
        {'id': 7, 'lat': 41.834344, 'lon': -87.623795},  # Farr Hall
        {'id': 8, 'lat': 41.834344, 'lon': -87.623795},  # Farr Hall
        {'id': 9, 'lat': 41.837866, 'lon': -87.624703},  # Kacek Hall
    ]

    # Create executor
    executor = ScenarioExecutor(
        trip_data=trip_data,
        station_info=station_info,
        vehicle_starts_template=vehicle_starts_template,
        ml_model_path=model_path,
        output_dir=output_dir,
    )

    # Generate scenarios based on args
    if args.all:
        scenarios = generate_all_scenarios()
        print(f"\nRunning FULL FACTORIAL: {len(scenarios)} scenarios × 3 methods")
    elif args.sweep:
        scenarios = generate_sweep_scenarios(args.sweep)
        print(f"\nRunning {args.sweep.upper()} SWEEP: "
              f"{len(scenarios)} scenarios × 3 methods")
    elif args.key:
        scenarios = generate_key_scenarios()
        print(f"\nRunning KEY SCENARIOS: {len(scenarios)} scenarios × 3 methods")
    else:
        # Single scenario from args
        sid = f"single_F{args.fleet}_E{int(args.epoch)}"
        scenarios = [ScenarioConfig(
            scenario_id=sid,
            fleet_size=args.fleet,
            epoch_minutes=args.epoch,
            weights=WEIGHT_PROFILES[args.weights],
            temperature=TEMP_PROFILES[args.temp],
        )]
        print(f"\nRunning SINGLE SCENARIO")

    # Execute
    t_total_start = _time.time()
    for i, config in enumerate(scenarios):
        print(f"\n[Scenario {i+1}/{len(scenarios)}]")
        executor.run_scenario(config)

    t_total = _time.time() - t_total_start
    print(f"\n{'='*70}")
    print(f"COMPLETED: {len(scenarios)} scenarios in {t_total/60:.1f} minutes")
    print(f"{'='*70}")

    # Save results
    executor.save_results()
    df_summary = executor.save_summary_csv()

    # Print quick summary
    print("\n--- QUICK SUMMARY ---")
    if len(df_summary) > 0:
        pivot = df_summary.pivot_table(
            index=['temperature', 'fleet_size'],
            columns='method',
            values='service_rate_pct',
            aggfunc='mean'
        )
        print("\nService Rate (%) by Temperature × Fleet Size:")
        print(pivot.round(1).to_string())


if __name__ == '__main__':
    main()