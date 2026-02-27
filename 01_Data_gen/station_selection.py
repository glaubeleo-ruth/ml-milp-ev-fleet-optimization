import numpy as np
import pandas as pd
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time as _time

# Script directory: default paths are relative to Data_gen so script works from any cwd
_SCRIPT_DIR = Path(__file__).resolve().parent

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print(
        "WARNING: Gurobi not installed.\n"
        "  Install: pip install gurobipy\n"
        "  Academic license (free): https://www.gurobi.com/academia/academic-program-and-licenses/"
    )


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for the EV-UFLP optimization."""
    
    # Battery parameters (GreenPower EV Star)
    battery_capacity_kWh: float = 118.0
    soc_min: float = 0.15              # minimum operating SoC
    soc_max: float = 0.95              # maximum charge SoC
    soc_threshold_to_charge: float = 0.30  # trigger charging at this SoC
    
    # Station constraints
    fixed_cost_per_station: float = 1.0   # gamma — set by sensitivity analysis
    max_distance_to_station_km: float = 2.0  # coverage constraint
    max_energy_to_station_kWh: float = None   # auto-computed from battery
    chargers_per_station: int = 2             # Level 2 chargers per station
    
    # Solver settings
    time_limit_sec: int = 60
    mip_gap: float = 0.001            # 0.1% optimality gap
    verbose: bool = False
    
    @property
    def usable_capacity(self) -> float:
        """Usable battery capacity between SoC limits (kWh)."""
        return self.battery_capacity_kWh * (self.soc_max - self.soc_min)
    
    @property
    def energy_budget_to_station(self) -> float:
        """
        Max energy a vehicle can spend reaching a station (kWh).
        
        Budget = (SoC_threshold - SoC_min) × capacity
        If vehicle decides to charge at 30% SoC, it must reach
        a station before hitting 15%.
        """
        return (self.soc_threshold_to_charge - self.soc_min) * self.battery_capacity_kWh


# =============================================================================
# DATA LOADING
# =============================================================================

def load_optimization_data(base_dir: Path) -> Dict:
    """
    Load all input files from Station_data.py outputs.
    
    Returns dict with: energy_matrix, distance_matrix, demand_df, candidates
    """
    energy_matrix = pd.read_csv(base_dir / 'energy_matrix.csv', index_col=0)
    distance_matrix = pd.read_csv(base_dir / 'distance_matrix.csv', index_col=0)
    demand_df = pd.read_csv(base_dir / 'demand_data.csv')
    candidates = pd.read_csv(base_dir / 'candidate_stations.csv')
    
    print(f"  Energy matrix: {energy_matrix.shape}")
    print(f"  Distance matrix: {distance_matrix.shape}")
    print(f"  Demand records: {len(demand_df)}")
    print(f"  Candidate stations: {len(candidates)}")
    
    return {
        'energy_matrix': energy_matrix,
        'distance_matrix': distance_matrix,
        'demand_df': demand_df,
        'candidates': candidates,
    }


def compute_demand_weights(demand_df: pd.DataFrame, poi_names: List[str]) -> List[float]:
    """
    Compute normalized demand weight for each POI.
    
    Weight = (origin_trips + dest_trips) / max_total_trips
    High-demand POIs get higher weights in the objective.
    """
    origin_trips = demand_df.groupby('origin')['trips'].sum()
    dest_trips = demand_df.groupby('destination')['trips'].sum()
    
    weights = []
    for name in poi_names:
        total = origin_trips.get(name, 0) + dest_trips.get(name, 0)
        weights.append(total)
    
    max_weight = max(weights) if max(weights) > 0 else 1
    weights = [w / max_weight for w in weights]
    
    return weights


# =============================================================================
# MODEL CONSTRUCTION
# =============================================================================

def build_ev_uflp(
    energy_matrix: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    demand_weights: List[float],
    config: OptimizationConfig
) -> Tuple[Optional['gp.Model'], List[str]]:
    """
    Build the EV-UFLP Gurobi model.
    
    Formulation:
        Sets:
            N = set of POIs (both demand points and candidate stations)
        
        Decision Variables:
            y_j ∈ {0,1}  — 1 if station opened at POI j
            x_ij ∈ {0,1} — 1 if demand point i assigned to station j
        
        Objective:
            min Σ_i Σ_j (w_i × e_ij × x_ij) + γ × Σ_j y_j
            
            where w_i = demand weight at POI i
                  e_ij = energy to travel from i to j (detour energy)
                  γ = infrastructure cost per station
        
        Constraints:
            (C1) Each POI assigned to exactly one station:
                 Σ_j x_ij = 1   ∀i
            
            (C2) Can only assign to open stations:
                 x_ij ≤ y_j     ∀i,j
            
            (C3) Battery feasibility — vehicle can reach assigned station:
                 x_ij = 0       if e_ij > energy_budget
            
            (C4) Distance coverage (optional):
                 x_ij = 0       if d_ij > max_distance
    """
    if not GUROBI_AVAILABLE:
        return None, []
    
    poi_names = list(energy_matrix.index)
    n = len(poi_names)
    E = energy_matrix.values
    D = distance_matrix.values
    
    # Energy budget for reaching a station
    energy_budget = config.energy_budget_to_station
    
    # Create model
    model = gp.Model("EV_UFLP")
    model.setParam('OutputFlag', 1 if config.verbose else 0)
    model.setParam('TimeLimit', config.time_limit_sec)
    model.setParam('MIPGap', config.mip_gap)
    
    # --- Decision Variables ---
    
    # y_j: open station at POI j
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    
    # x_ij: assign demand point i to station j
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    
    # --- Objective ---
    # min Σ_i Σ_j (w_i × e_ij × x_ij) + γ × Σ_j y_j
    
    detour_cost = gp.quicksum(
        demand_weights[i] * E[i, j] * x[i, j]
        for i in range(n) for j in range(n)
        if E[i, j] <= energy_budget  # only feasible pairs
    )
    
    infra_cost = config.fixed_cost_per_station * gp.quicksum(
        y[j] for j in range(n)
    )
    
    model.setObjective(detour_cost + infra_cost, GRB.MINIMIZE)
    
    # --- Constraints ---
    
    # (C1) Each POI assigned to exactly one station
    for i in range(n):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(n)) == 1,
            name=f"assign_{i}"
        )
    
    # (C2) Can only assign to open stations
    for i in range(n):
        for j in range(n):
            model.addConstr(x[i, j] <= y[j], name=f"open_{i}_{j}")
    
    # (C3) Battery feasibility — fix infeasible assignments to 0
    for i in range(n):
        for j in range(n):
            if E[i, j] > energy_budget:
                model.addConstr(x[i, j] == 0, name=f"battery_{i}_{j}")
    
    # (C4) Distance coverage constraint (optional)
    if config.max_distance_to_station_km is not None:
        for i in range(n):
            for j in range(n):
                if i != j and D[i, j] > config.max_distance_to_station_km:
                    model.addConstr(x[i, j] == 0, name=f"dist_{i}_{j}")
    
    model.update()
    
    return model, poi_names


# =============================================================================
# SOLVING
# =============================================================================

def solve_model(
    model: 'gp.Model',
    poi_names: List[str],
    config: OptimizationConfig
) -> Optional[Dict]:
    """
    Solve the UFLP model and extract results.
    
    Returns dict with: status, objective_value, n_stations,
    selected_stations, assignments, solve_time_sec
    """
    if model is None:
        return None
    
    n = len(poi_names)
    
    start = _time.time()
    model.optimize()
    solve_time = _time.time() - start
    
    if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        return {
            'status': 'infeasible' if model.status == GRB.INFEASIBLE else f'status_{model.status}',
            'n_stations': 0,
        }
    
    # Extract station selections
    y_vals = {j: model.getVarByName(f"y[{j}]").X for j in range(n)}
    selected_indices = [j for j in range(n) if y_vals[j] > 0.5]
    selected_stations = [poi_names[j] for j in selected_indices]
    
    # Extract assignments
    assignments = {}
    for i in range(n):
        for j in range(n):
            var = model.getVarByName(f"x[{i},{j}]")
            if var.X > 0.5:
                assignments[poi_names[i]] = poi_names[j]
                break
    
    return {
        'status': 'optimal' if model.status == GRB.OPTIMAL else 'time_limit',
        'objective_value': model.ObjVal,
        'n_stations': len(selected_stations),
        'selected_stations': selected_stations,
        'selected_indices': selected_indices,
        'assignments': assignments,
        'solve_time_sec': solve_time,
        'mip_gap': model.MIPGap,
    }


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def run_sensitivity_analysis(
    energy_matrix: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    demand_weights: List[float],
    config: OptimizationConfig,
    gamma_values: List[float] = None
) -> pd.DataFrame:
    """
    Sweep gamma values to trace the Pareto frontier.
    
    This reveals the Pareto frontier: how many stations the optimizer selects
    at different cost-benefit tradeoff points. The "elbow" of the curve
    indicates the natural optimal number of stations.
    
    Parameters
    ----------
    gamma_values : list of float
        Infrastructure cost values to test. If None, auto-generates a range.
    
    Returns
    -------
    pd.DataFrame
        Columns: gamma, n_stations, objective, infra_cost, detour_energy,
                 selected_stations, solve_time
    """
    if gamma_values is None:
        # Auto-generate range based on energy scale
        E = energy_matrix.values
        avg_energy = E[E > 0].mean()
        # Range from very cheap (many stations) to very expensive (few stations)
        gamma_values = [
            avg_energy * mult for mult in
            [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        ]
    
    results = []
    
    for gamma in gamma_values:
        config_copy = OptimizationConfig(
            battery_capacity_kWh=config.battery_capacity_kWh,
            soc_min=config.soc_min,
            soc_max=config.soc_max,
            fixed_cost_per_station=gamma,
            max_distance_to_station_km=config.max_distance_to_station_km,
            max_energy_to_station_kWh=config.max_energy_to_station_kWh,
            soc_threshold_to_charge=config.soc_threshold_to_charge,
            time_limit_sec=config.time_limit_sec,
            mip_gap=config.mip_gap,
            verbose=False
        )
        
        model, poi_names = build_ev_uflp(
            energy_matrix, distance_matrix, demand_weights, config_copy
        )
        
        if model is None:
            continue
        
        result = solve_model(model, poi_names, config_copy)
        
        if result and result['n_stations'] > 0:
            # Decompose objective
            infra_cost = gamma * result['n_stations']
            detour_energy = result['objective_value'] - infra_cost
            
            results.append({
                'gamma': gamma,
                'n_stations': result['n_stations'],
                'objective': result['objective_value'],
                'infra_cost': infra_cost,
                'detour_energy': detour_energy,
                'selected_stations': ', '.join(result['selected_stations']),
                'solve_time_sec': result.get('solve_time_sec', 0)
            })
            
            print(f"  γ={gamma:.4f} → {result['n_stations']} stations: "
                  f"{result['selected_stations']}")
    
    return pd.DataFrame(results)


def find_elbow_point(sensitivity_df: pd.DataFrame) -> Dict:
    """
    Find the elbow point in the sensitivity analysis.
    
    The elbow is where adding another station yields diminishing returns
    in detour energy reduction. Uses the maximum-distance-from-line method:
    draw a line from the first to last point on the (n_stations, detour_energy)
    curve, then find the point farthest from that line.
    
    Returns dict with: gamma, n_stations, selected_stations
    """
    # Deduplicate by n_stations (keep lowest gamma per station count)
    deduped = sensitivity_df.sort_values('gamma').drop_duplicates(
        subset='n_stations', keep='first'
    ).sort_values('n_stations', ascending=False)
    
    if len(deduped) < 3:
        # Not enough points — return the middle one
        mid = len(deduped) // 2
        row = deduped.iloc[mid]
        return {
            'gamma': row['gamma'],
            'n_stations': int(row['n_stations']),
            'selected_stations': row['selected_stations'],
        }
    
    # Maximum distance from line method
    x = deduped['n_stations'].values.astype(float)
    y = deduped['detour_energy'].values.astype(float)
    
    # Line from first point (most stations) to last point (fewest stations)
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    
    # Normalize axes to [0,1] for fair distance comparison
    x_range = x[0] - x[-1] if x[0] != x[-1] else 1
    y_range = y[-1] - y[0] if y[-1] != y[0] else 1
    
    max_dist = 0
    elbow_idx = len(deduped) // 2  # default to middle
    
    for i in range(len(x)):
        # Normalized point
        px = (x[i] - x[-1]) / x_range
        py = (y[i] - y[0]) / y_range
        
        # Normalized line endpoints
        p1n = np.array([(x[0] - x[-1]) / x_range, (y[0] - y[0]) / y_range])
        p2n = np.array([(x[-1] - x[-1]) / x_range, (y[-1] - y[0]) / y_range])
        
        # Distance from point to line
        line_vec = p2n - p1n
        point_vec = np.array([px, py]) - p1n
        
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            continue
        
        # Perpendicular distance
        dist = abs(np.cross(line_vec, point_vec)) / line_len
        
        if dist > max_dist:
            max_dist = dist
            elbow_idx = i
    
    row = deduped.iloc[elbow_idx]
    
    return {
        'gamma': row['gamma'],
        'n_stations': int(row['n_stations']),
        'selected_stations': row['selected_stations'],
    }


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def generate_outputs(
    result: Dict,
    candidates: pd.DataFrame,
    energy_matrix: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    config: OptimizationConfig,
    output_dir: Path
):
    """
    Save all output files for downstream consumption.
    
    Outputs:
        - selected_stations.csv:  station locations with charger count
        - station_assignment.csv: POI-to-station mapping with distances
        - station_selection_report.json: summary for MILP consumption
    """
    # --- selected_stations.csv ---
    station_records = []
    for station_name in result['selected_stations']:
        # Find in candidates
        match = candidates[candidates['poi_name'] == station_name]
        if len(match) > 0:
            row = match.iloc[0]
            station_records.append({
                'station_name': station_name,
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'category': row['category'],
                'charger_count': config.chargers_per_station,
                'demand_weight': row.get('demand_weight', 0),
            })
        else:
            # Fallback: look up from energy matrix index
            station_records.append({
                'station_name': station_name,
                'latitude': None,
                'longitude': None,
                'category': None,
                'charger_count': config.chargers_per_station,
                'demand_weight': 0,
            })
    
    stations_df = pd.DataFrame(station_records)
    stations_df.to_csv(output_dir / 'selected_stations.csv', index=False)
    
    # --- station_assignment.csv ---
    assignment_records = []
    for poi, station in result['assignments'].items():
        # Look up distance and energy
        try:
            dist = distance_matrix.loc[poi, station]
            energy = energy_matrix.loc[poi, station]
        except KeyError:
            dist = 0
            energy = 0
        
        assignment_records.append({
            'poi': poi,
            'assigned_station': station,
            'distance_km': round(dist, 4),
            'energy_kWh': round(energy, 4),
            'is_station': poi == station,
        })
    
    assignment_df = pd.DataFrame(assignment_records)
    assignment_df.to_csv(output_dir / 'station_assignment.csv', index=False)
    
    # --- station_selection_report.json ---
    report = {
        'model': 'EV-UFLP',
        'solver': 'Gurobi',
        'status': result['status'],
        'objective_value': result['objective_value'],
        'n_stations': result['n_stations'],
        'selected_stations': result['selected_stations'],
        'gamma': config.fixed_cost_per_station,
        'mip_gap': result.get('mip_gap', 0),
        'solve_time_sec': result.get('solve_time_sec', 0),
        'config': {
            'battery_capacity_kWh': config.battery_capacity_kWh,
            'soc_min': config.soc_min,
            'soc_max': config.soc_max,
            'soc_threshold_to_charge': config.soc_threshold_to_charge,
            'energy_budget_to_station_kWh': config.energy_budget_to_station,
            'chargers_per_station': config.chargers_per_station,
            'max_distance_to_station_km': config.max_distance_to_station_km,
        },
        'assignments': result['assignments'],
    }
    
    with open(output_dir / 'station_selection_report.json', 'w') as f:
        json.dump(report, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def main(data_dir: str = None, output_dir: str = None) -> Optional[Dict]:
    """
    Run the full station selection pipeline.
    
    1. Load data from Station_data.py outputs
    2. Compute demand weights
    3. Run sensitivity analysis (sweep gamma)
    4. Find elbow point (optimal number of stations)
    5. Run final optimization at elbow gamma
    6. Save outputs
    
    Parameters
    ----------
    data_dir : str, optional
        Directory containing Station_data.py output files (energy_matrix.csv, etc.).
        Default: Data_gen/Outputs (same folder where station_data.py writes).
    output_dir : str, optional
        Directory for output files. Defaults to data_dir.
    """
    print("=" * 60)
    print("EV Charging Station Selection Optimizer")
    print("=" * 60)
    print("Model: EV-UFLP (Uncapacitated Facility Location Problem)")
    print("Solver: Gurobi")
    
    if not GUROBI_AVAILABLE:
        print("\nERROR: Gurobi is required but not installed.")
        print("  1. Install: pip install gurobipy")
        print("  2. Get a free academic license: https://www.gurobi.com/academia/academic-program-and-licenses/")
        print("  3. For pip-only install, use a WLS license (no grbgetkey):")
        print("     - Request 'Academic WLS' in the portal, then go to https://license.gurobi.com/")
        print("     - Create an API key and save the downloaded gurobi.lic to ~/gurobi.lic (or set GRB_LICENSE_FILE)")
        return None
    
    if data_dir is None:
        base_dir = _SCRIPT_DIR / 'Outputs'
    else:
        base_dir = Path(data_dir)
    out_dir = Path(output_dir) if output_dir else base_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Load data ---
    print("\n" + "-" * 40)
    print("LOADING DATA (from Station_data.py outputs)")
    print("-" * 40)
    
    data = load_optimization_data(base_dir)
    
    # --- Compute demand weights ---
    poi_names = list(data['energy_matrix'].index)
    demand_weights = compute_demand_weights(data['demand_df'], poi_names)
    
    print(f"\n  Demand weights (top 5):")
    weight_ranking = sorted(zip(poi_names, demand_weights), key=lambda x: -x[1])
    for name, w in weight_ranking[:5]:
        print(f"    {name}: {w:.3f}")
    
    # --- Configuration ---
    config = OptimizationConfig()
    
    print(f"\n  Battery capacity: {config.battery_capacity_kWh} kWh")
    print(f"  Usable capacity: {config.usable_capacity:.1f} kWh")
    print(f"  Energy budget to reach station: {config.energy_budget_to_station:.2f} kWh")
    print(f"  Max distance to station: {config.max_distance_to_station_km} km")
    
    # --- Sensitivity analysis ---
    print("\n" + "-" * 40)
    print("SENSITIVITY ANALYSIS (varying gamma)")
    print("-" * 40)
    print("  Finding optimal cost-benefit tradeoff...\n")
    
    sensitivity_df = run_sensitivity_analysis(
        data['energy_matrix'],
        data['distance_matrix'],
        demand_weights,
        config
    )
    
    if len(sensitivity_df) == 0:
        print("ERROR: No feasible solutions found in sensitivity analysis.")
        return None
    
    # Save sensitivity results
    sensitivity_df.to_csv(out_dir / "sensitivity_analysis.csv", index=False)
    
    # Find elbow point
    elbow = find_elbow_point(sensitivity_df)
    
    print(f"\n  ELBOW POINT: {elbow['n_stations']} stations")
    print(f"  Recommended gamma: {elbow['gamma']:.4f}")
    print(f"  Stations: {elbow['selected_stations']}")
    
    # --- Final optimization at elbow point ---
    print("\n" + "-" * 40)
    print("FINAL OPTIMIZATION (at elbow gamma)")
    print("-" * 40)
    
    config.fixed_cost_per_station = elbow['gamma']
    config.verbose = True
    
    model, poi_names = build_ev_uflp(
        data['energy_matrix'],
        data['distance_matrix'],
        demand_weights,
        config
    )
    
    result = solve_model(model, poi_names, config)
    
    if result and result['n_stations'] > 0:
        print(f"\n  Status: {result['status']}")
        print(f"  Stations selected: {result['n_stations']}")
        print(f"  Objective value: {result['objective_value']:.4f}")
        print(f"  Solve time: {result.get('solve_time_sec', 0):.3f} sec")
        
        print(f"\n  SELECTED STATIONS:")
        for station in result['selected_stations']:
            assigned = [p for p, s in result['assignments'].items() if s == station]
            print(f"    📍 {station} — serves {len(assigned)} POIs")
        
        print(f"\n  POI ASSIGNMENTS:")
        for poi, station in sorted(result['assignments'].items()):
            marker = " ⚡" if poi == station else ""
            print(f"    {poi} → {station}{marker}")
        
        # --- Save outputs ---
        print("\n" + "-" * 40)
        print("SAVING OUTPUTS")
        print("-" * 40)
        
        generate_outputs(
            result,
            data['candidates'],
            data['energy_matrix'],
            data['distance_matrix'],
            config,
            out_dir
        )
        
        print("  selected_stations.csv")
        print("  station_assignment.csv")
        print("  sensitivity_analysis.csv")
        print("  station_selection_report.json")
        print("\n✓ Station selection complete")
        
        return {
            'result': result,
            'sensitivity': sensitivity_df,
            'elbow': elbow,
            'config': config
        }
    else:
        print(f"\n  ERROR: Optimization failed — {result.get('status', 'unknown')}")
        return None


if __name__ == "__main__":
    output = main()