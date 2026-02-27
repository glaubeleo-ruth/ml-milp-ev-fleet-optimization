import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
from pathlib import Path

# Script directory: paths are relative to Data_gen so the script works from any cwd
_SCRIPT_DIR = Path(__file__).resolve().parent


# =============================================================================
# VEHICLE PARAMETERS (GreenPower EV Star MY20 — Minimal Set)
# =============================================================================

@dataclass
class VehicleParams:
    """
    GreenPower EV Star parameters — minimal set for energy calculation.
    
    Source: GreenPower Motor Company MY20 EV Star spec sheet.
    """
    
    # Mass
    mass_curb: float = 4500.0           # kg (curb weight)
    mass_gvwr: float = 6500.0           # kg (gross vehicle weight rating)
    mass_passenger: float = 70.0        # kg (average passenger mass)
    mass_rotational_factor: float = 1.05  # accounts for rotating components
    
    # Aerodynamics
    Cd: float = 0.65                    # drag coefficient (flat-front shuttle)
    frontal_area: float = 4.65          # m² (estimated frontal cross-section)
    
    # Rolling resistance
    Crr: float = 0.010                  # standard for bus tires on pavement
    
    # Efficiency
    eta_drivetrain: float = 0.85        # combined motor + inverter average
    eta_regen: float = 0.65             # regenerative braking efficiency
    eta_battery: float = 0.95           # battery round-trip efficiency
    
    # Battery
    battery_capacity: float = 118.0     # kWh (total)
    soc_min: float = 0.15              # minimum operating SoC
    soc_max: float = 0.95              # maximum SoC (charge limit)
    
    # Auxiliary power
    aux_power: float = 2.0              # kW (HVAC + lights + electronics)
    
    # Physical constants
    air_density: float = 1.225          # kg/m³
    gravity: float = 9.81              # m/s²
    
    @property
    def CdA(self) -> float:
        """Drag area product."""
        return self.Cd * self.frontal_area
    
    @property
    def usable_capacity(self) -> float:
        """Usable battery capacity between SoC limits (kWh)."""
        return self.battery_capacity * (self.soc_max - self.soc_min)


# =============================================================================
# ENERGY MODEL (Simplified Longitudinal Vehicle Dynamics)
# =============================================================================

class EnergyModel:
    """
    Simplified physics-based energy consumption model.
    
    Core equation (Newton's second law on a road):
        F_total = F_rolling + F_aero + F_grade + F_accel
    
    Energy is converted from mechanical (J) to electrical (kWh)
    through the drivetrain efficiency chain.
    
    Reference: Fiori et al. (2016) — Power-based EV energy consumption model
    """
    
    def __init__(self, params: VehicleParams = None):
        self.params = params or VehicleParams()
    
    def compute_trip_energy(
        self,
        distance_km: float,
        avg_speed_kmh: float = 25.0,
        n_passengers: int = 5,
        n_stops: int = 0,
        grade_avg: float = 0.0
    ) -> float:
        """
        Compute energy consumption for a trip.
        
        Parameters
        ----------
        distance_km : float
            Road distance in km.
        avg_speed_kmh : float
            Average speed in km/h.
        n_passengers : int
            Number of passengers onboard.
        n_stops : int
            Number of intermediate stops (for acceleration energy).
        grade_avg : float
            Average road grade (decimal, e.g., 0.02 = 2%).
        
        Returns
        -------
        float
            Energy consumed in kWh.
        """
        p = self.params
        
        # Total vehicle mass
        total_mass = p.mass_curb + n_passengers * p.mass_passenger
        effective_mass = total_mass * p.mass_rotational_factor
        
        # Convert units
        distance_m = distance_km * 1000
        speed_ms = avg_speed_kmh / 3.6
        
        if distance_m <= 0 or speed_ms <= 0:
            return 0.0
        
        # ---- Force components (N) ----
        
        # Rolling resistance: F_roll = Crr × m × g
        F_rolling = p.Crr * total_mass * p.gravity
        
        # Aerodynamic drag: F_aero = 0.5 × ρ × CdA × v²
        F_aero = 0.5 * p.air_density * p.CdA * speed_ms**2
        
        # Grade resistance: F_grade = m × g × sin(θ) ≈ m × g × θ
        F_grade = total_mass * p.gravity * grade_avg
        
        # ---- Energy components (J) ----
        
        # Traction energy = F_total × distance
        E_traction = (F_rolling + F_aero + F_grade) * distance_m
        
        # Acceleration energy at stops: E_accel = 0.5 × m × v² per stop
        # Only the fraction not recovered by regen braking is lost
        n_accel_events = max(1, n_stops + 1)  # at least 1 for trip start
        E_accel_per_stop = 0.5 * effective_mass * speed_ms**2
        E_accel = n_accel_events * E_accel_per_stop * (1 - p.eta_regen)
        
        # Total mechanical energy (J)
        E_mech = E_traction + E_accel
        
        # ---- Convert to electrical energy (kWh) ----
        
        # Divide by drivetrain and battery efficiency
        E_electrical_j = E_mech / (p.eta_drivetrain * p.eta_battery)
        
        # Convert J to kWh
        E_electrical_kwh = E_electrical_j / 3_600_000
        
        # Add auxiliary loads (HVAC, lights, electronics)
        trip_duration_h = distance_km / avg_speed_kmh
        E_aux_kwh = p.aux_power * trip_duration_h
        
        return E_electrical_kwh + E_aux_kwh


# =============================================================================
# DISTANCE CALCULATIONS
# =============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two points on Earth (km).
    
    Standard haversine formula for geodesic distance.
    """
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def road_distance(straight_line_km: float, circuity_factor: float = 1.3) -> float:
    """
    Estimate road distance from straight-line distance.
    
    Circuity factor 1.3 is well-established for urban grids
    (Ballou et al., 2002). Chicago's grid layout makes this accurate.
    """
    return straight_line_km * circuity_factor


# =============================================================================
# DEMAND GENERATION
# =============================================================================

@dataclass
class DemandPattern:
    """
    Time-of-day demand multipliers by OD category.
    
    Encodes realistic campus travel behavior based on
    university transit planning survey patterns.
    """
    
    @staticmethod
    def get_multiplier(hour: int, origin_cat: str, dest_cat: str) -> float:
        """
        Get demand multiplier based on time of day and OD categories.
        
        Patterns:
            Residence → Academic: peaks 7-9 AM (morning commute)
            Academic → Residence: peaks 4-6 PM (afternoon commute)
            → Dining: peaks at meal times (11-1 PM, 5-7 PM)
            → Library: sustained 10 AM - 10 PM
            → Recreation: afternoon/evening (3-9 PM)
        """
        # Morning commute: Residence → Academic
        if origin_cat == 'Residence' and dest_cat == 'Academic':
            if 7 <= hour <= 9:
                return 1.0
            elif 10 <= hour <= 11:
                return 0.3
            else:
                return 0.1
        
        # Afternoon commute: Academic → Residence
        elif origin_cat == 'Academic' and dest_cat == 'Residence':
            if 16 <= hour <= 18:
                return 1.0
            elif 12 <= hour <= 14:
                return 0.4
            elif 19 <= hour <= 21:
                return 0.3
            else:
                return 0.1
        
        # Meal times: → Dining
        elif dest_cat == 'Dining':
            if 11 <= hour <= 13 or 17 <= hour <= 19:
                return 1.0
            elif 7 <= hour <= 9:
                return 0.5
            else:
                return 0.2
        
        # Study hours: → Library
        elif dest_cat == 'Library':
            if 10 <= hour <= 22:
                return 0.6
            else:
                return 0.1
        
        # Afternoon/evening: → Recreation
        elif dest_cat == 'Recreation':
            if 15 <= hour <= 21:
                return 0.7
            else:
                return 0.2
        
        # Default pattern
        else:
            if 8 <= hour <= 18:
                return 0.5
            else:
                return 0.2


def generate_daily_demand(
    pois: pd.DataFrame,
    base_trips_per_hour: float = 8.0,
    operating_hours: Tuple[int, int] = (7, 23),
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic daily trip demand using category-based
    temporal patterns and Poisson sampling.
    
    Three-layer demand model:
        Layer 1: Temporal patterns by OD category
        Layer 2: Priority weighting — high-priority POIs generate more traffic
        Layer 3: Poisson sampling for realistic integer trip counts
    
    Parameters
    ----------
    pois : pd.DataFrame
        POI data with columns: poi_name, latitude, longitude, category, priority
    base_trips_per_hour : float
        Base expected trips per hour per OD pair (before multipliers).
    operating_hours : tuple
        (start_hour, end_hour) of shuttle operations.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    pd.DataFrame
        Columns: origin, destination, hour, multiplier, expected_trips, trips
    """
    np.random.seed(seed)
    
    records = []
    start_h, end_h = operating_hours
    
    for _, origin in pois.iterrows():
        for _, dest in pois.iterrows():
            if origin['poi_name'] == dest['poi_name']:
                continue
            
            for hour in range(start_h, end_h):
                # Layer 1: Category-based temporal multiplier
                multiplier = DemandPattern.get_multiplier(
                    hour, origin['category'], dest['category']
                )
                
                # Layer 2: Priority weighting
                # High priority (1) → weight 4/4 = 1.0
                # Low priority (4) → weight 1/4 = 0.25
                priority_weight = (
                    (5 - origin['priority']) * (5 - dest['priority']) / 16
                )
                
                # Expected trips (Poisson lambda)
                expected = base_trips_per_hour * multiplier * priority_weight
                
                # Layer 3: Poisson sampling
                trips = np.random.poisson(expected) if expected > 0 else 0
                
                if trips > 0 or expected > 0:
                    records.append({
                        'origin': origin['poi_name'],
                        'destination': dest['poi_name'],
                        'hour': hour,
                        'origin_category': origin['category'],
                        'dest_category': dest['category'],
                        'multiplier': round(multiplier, 3),
                        'priority_weight': round(priority_weight, 4),
                        'expected_trips': round(expected, 3),
                        'trips': trips,
                    })
    
    return pd.DataFrame(records)


# =============================================================================
# MATRIX COMPUTATION
# =============================================================================

def compute_distance_matrix(pois: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 16×16 pairwise road distance matrix (km).
    
    Uses haversine for straight-line distance, then applies
    circuity factor of 1.3 for road distance estimation.
    """
    n = len(pois)
    names = pois['poi_name'].tolist()
    
    matrix = np.zeros((n, n))
    
    for i, (_, poi1) in enumerate(pois.iterrows()):
        for j, (_, poi2) in enumerate(pois.iterrows()):
            if i != j:
                dist_straight = haversine_distance(
                    poi1['latitude'], poi1['longitude'],
                    poi2['latitude'], poi2['longitude']
                )
                matrix[i, j] = road_distance(dist_straight)
    
    return pd.DataFrame(matrix, index=names, columns=names)


def compute_energy_matrix(
    pois: pd.DataFrame,
    avg_speed_kmh: float = 20.0,
    avg_passengers: int = 5
) -> pd.DataFrame:
    """
    Compute 16×16 pairwise energy consumption matrix (kWh).
    
    Uses the physics-based energy model with default campus
    operating conditions: 20 km/h average speed, 5 passengers.
    """
    energy_model = EnergyModel()
    
    n = len(pois)
    names = pois['poi_name'].tolist()
    
    matrix = np.zeros((n, n))
    
    for i, (_, poi1) in enumerate(pois.iterrows()):
        for j, (_, poi2) in enumerate(pois.iterrows()):
            if i != j:
                dist_straight = haversine_distance(
                    poi1['latitude'], poi1['longitude'],
                    poi2['latitude'], poi2['longitude']
                )
                dist_road = road_distance(dist_straight)
                
                matrix[i, j] = energy_model.compute_trip_energy(
                    distance_km=dist_road,
                    avg_speed_kmh=avg_speed_kmh,
                    n_passengers=avg_passengers,
                    n_stops=0
                )
    
    return pd.DataFrame(matrix, index=names, columns=names)


# =============================================================================
# CANDIDATE STATION LOCATIONS
# =============================================================================

def identify_candidate_stations(
    pois: pd.DataFrame,
    demand_df: pd.DataFrame,
    n_candidates: int = None
) -> pd.DataFrame:
    """
    Identify candidate charging station locations.
    
    Ranks POIs by total trip volume (as origin + destination).
    Logic: stations should go where shuttles spend the most time,
    which correlates with high-demand locations.
    """
    # Aggregate demand at each POI (as origin or destination)
    origin_demand = demand_df.groupby('origin')['trips'].sum()
    dest_demand = demand_df.groupby('destination')['trips'].sum()
    
    # Total demand at each POI
    poi_demand = pd.DataFrame({
        'poi_name': pois['poi_name'],
        'latitude': pois['latitude'],
        'longitude': pois['longitude'],
        'category': pois['category'],
        'priority': pois['priority']
    })
    
    poi_demand['origin_trips'] = poi_demand['poi_name'].map(origin_demand).fillna(0)
    poi_demand['dest_trips'] = poi_demand['poi_name'].map(dest_demand).fillna(0)
    poi_demand['total_trips'] = poi_demand['origin_trips'] + poi_demand['dest_trips']
    
    # Normalize to weight (0-1)
    max_trips = poi_demand['total_trips'].max()
    poi_demand['demand_weight'] = poi_demand['total_trips'] / max_trips if max_trips > 0 else 0
    
    # Rank by demand
    poi_demand = poi_demand.sort_values('total_trips', ascending=False)
    
    if n_candidates:
        poi_demand = poi_demand.head(n_candidates)
    
    return poi_demand.reset_index(drop=True)


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def compute_summary_stats(
    pois: pd.DataFrame,
    demand_df: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    energy_matrix: pd.DataFrame
) -> Dict:
    """
    Compute summary statistics for optimization.
    
    Includes fleet size estimate using Little's Law reasoning:
    if peak hour has P trips and each takes ~15 min, need >= P/4 vehicles.
    """
    total_daily_trips = demand_df['trips'].sum()
    
    # Peak hour
    hourly_trips = demand_df.groupby('hour')['trips'].sum()
    peak_hour = hourly_trips.idxmax()
    peak_trips = hourly_trips.max()
    
    # Distance stats (non-zero entries)
    dist_values = distance_matrix.values[distance_matrix.values > 0]
    energy_values = energy_matrix.values[energy_matrix.values > 0]
    
    # Fleet size estimate (Little's Law heuristic)
    avg_trip_duration_min = 15  # campus trips ~15 min including pickup
    vehicles_needed = int(np.ceil(peak_trips / (60 / avg_trip_duration_min)))
    
    # Daily energy estimate
    avg_energy_per_trip = energy_values.mean()
    daily_energy = total_daily_trips * avg_energy_per_trip
    
    return {
        'n_pois': len(pois),
        'n_categories': len(pois['category'].unique()),
        'categories': pois['category'].unique().tolist(),
        'total_daily_trips': int(total_daily_trips),
        'peak_hour': int(peak_hour),
        'peak_hour_trips': int(peak_trips),
        'operating_hours': '7:00 - 23:00',
        'avg_distance_km': round(float(dist_values.mean()), 3),
        'max_distance_km': round(float(dist_values.max()), 3),
        'min_distance_km': round(float(dist_values.min()), 3),
        'avg_energy_kwh': round(float(avg_energy_per_trip), 4),
        'max_energy_kwh': round(float(energy_values.max()), 4),
        'daily_energy_kwh': round(float(daily_energy), 1),
        'estimated_fleet_size': vehicles_needed,
        'fleet_size_basis': f'{peak_trips} peak trips / {60 // avg_trip_duration_min} trips per vehicle per hour',
    }


# =============================================================================
# MAIN
# =============================================================================

def main(
    pois_path: str = None,
    output_dir: str = None
) -> Dict:
    """
    Run the full data generation pipeline.
    
    Parameters
    ----------
    pois_path : str, optional
        Path to selected_pois.csv. Default: Data_gen/selected_pois.csv (relative to script).
    output_dir : str, optional
        Directory for output files. Default: Data_gen/Outputs (relative to script).
    """
    if pois_path is None:
        pois_path = _SCRIPT_DIR / 'selected_pois.csv'
    else:
        pois_path = Path(pois_path)
    if output_dir is None:
        output_dir = _SCRIPT_DIR / 'Outputs'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("EV Shuttle Data Generation")
    print("=" * 60)
    
    # Load POIs
    pois = pd.read_csv(pois_path)
    print(f"\nLoaded {len(pois)} POIs")
    print(f"Categories: {pois['category'].unique().tolist()}")
    
    # Generate demand
    print("\nGenerating daily demand...")
    demand_df = generate_daily_demand(pois, base_trips_per_hour=8.0)
    print(f"Generated {len(demand_df)} OD-hour combinations")
    print(f"Total daily trips: {demand_df['trips'].sum()}")
    
    # Compute matrices
    print("\nComputing distance matrix...")
    distance_matrix = compute_distance_matrix(pois)
    
    print("Computing energy matrix...")
    energy_matrix = compute_energy_matrix(pois)
    
    # Identify candidate stations
    print("\nIdentifying candidate station locations...")
    candidates = identify_candidate_stations(pois, demand_df)
    
    # Compute summary
    print("\nComputing summary statistics...")
    summary = compute_summary_stats(pois, demand_df, distance_matrix, energy_matrix)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Print top candidate stations
    print("\n" + "=" * 60)
    print("TOP CANDIDATE STATION LOCATIONS (by demand)")
    print("=" * 60)
    print(candidates[['poi_name', 'category', 'total_trips']].to_string(index=False))
    
    # Save outputs
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)
    
    demand_path = output_dir / 'demand_data.csv'
    demand_df.to_csv(demand_path, index=False)
    print(f"  demand_data.csv — {len(demand_df)} OD-hour records")
    
    dist_path = output_dir / 'distance_matrix.csv'
    distance_matrix.to_csv(dist_path)
    print(f"  distance_matrix.csv — {len(pois)}×{len(pois)} pairwise distances (km)")
    
    energy_path = output_dir / 'energy_matrix.csv'
    energy_matrix.to_csv(energy_path)
    print(f"  energy_matrix.csv — {len(pois)}×{len(pois)} pairwise energy (kWh)")
    
    cand_path = output_dir / 'candidate_stations.csv'
    candidates.to_csv(cand_path, index=False)
    print(f"  candidate_stations.csv — {len(candidates)} ranked candidates")
    
    summary_path = output_dir / 'summary_stats.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  summary_stats.json — daily summary")
    
    print("\n✓ Data generation complete")
    
    return {
        'pois': pois,
        'demand': demand_df,
        'distance_matrix': distance_matrix,
        'energy_matrix': energy_matrix,
        'candidates': candidates,
        'summary': summary
    }


if __name__ == "__main__":
    data = main()