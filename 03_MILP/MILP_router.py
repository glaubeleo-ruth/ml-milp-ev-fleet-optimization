import numpy as np
import pickle
import json
import time as _time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import pulp


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class VehicleState:
    """Snapshot of a vehicle at the current decision epoch."""
    vehicle_id: int
    lat: float
    lon: float
    soc: float              # 0–1
    status: str             # 'idle', 'charging', 'en_route', etc.
    km_today: float = 0.0


@dataclass
class TripRequest:
    """A pending trip request waiting for assignment."""
    trip_id: int
    origin: str
    origin_lat: float
    origin_lon: float
    destination: str
    dest_lat: float
    dest_lon: float
    distance_km: float
    n_passengers: int
    request_time: float     # seconds since midnight
    # Features needed for ML prediction
    avg_speed_kmh: float = 20.0
    temperature_C: float = 4.0
    grade_avg: float = 0.0
    n_stops: int = 1
    hour: int = 12


@dataclass
class StationState:
    """Snapshot of a charging station."""
    name: str
    lat: float
    lon: float
    total_chargers: int = 2
    occupied_chargers: int = 0

    @property
    def available_chargers(self) -> int:
        return self.total_chargers - self.occupied_chargers


@dataclass
class VehicleParams:
    """Vehicle specifications (GreenPower EV Star)."""
    battery_capacity: float = 118.0     # kWh
    curb_weight: float = 6577.0         # kg
    charge_rate: float = 19.2           # kW (Level 2)


# =============================================================================
# MILP SOLVER
# =============================================================================

class FleetMILP:
    """
    MILP formulation for joint dispatch + charging optimization.

    Usage:
        milp = FleetMILP(vehicle_params, ml_model_path)
        solution = milp.solve(vehicles, trips, stations, current_time)
    """

    def __init__(
        self,
        vehicle_params: VehicleParams,
        ml_model_path: Optional[str] = None,
        # Objective weights
        alpha: float = 1.0,     # energy weight
        beta: float = 0.01,     # wait time weight (sec → normalize)
        gamma: float = 10.0,    # unserved penalty
        # Thresholds
        soc_min: float = 0.10,          # absolute minimum SoC
        soc_charge_trigger: float = 0.30, # send to charge if below this
        # Solver
        time_limit_sec: float = 30.0,
        mip_gap: float = 0.01,
    ):
        self.params = vehicle_params
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.soc_min = soc_min
        self.soc_charge_trigger = soc_charge_trigger
        self.time_limit_sec = time_limit_sec
        self.mip_gap = mip_gap

        # Load ML model
        self.ml_model = None
        self.ml_features = None
        if ml_model_path and Path(ml_model_path).exists():
            with open(ml_model_path, 'rb') as f:
                bundle = pickle.load(f)
            self.ml_model = bundle['model']
            self.ml_features = bundle['features']
            print(f"  Loaded ML model: {bundle.get('model_name', 'unknown')}")

    # -----------------------------------------------------------------
    # PARAMETER COMPUTATION
    # -----------------------------------------------------------------

    def _haversine_km(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Great-circle distance between two points (km)."""
        R = 6371.0
        rlat1, rlon1 = np.radians(lat1), np.radians(lon1)
        rlat2, rlon2 = np.radians(lat2), np.radians(lon2)
        dlat = rlat2 - rlat1
        dlon = rlon2 - rlon1
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2)
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    def _predict_energy(
        self,
        distance_km: float,
        n_passengers: int,
        avg_speed_kmh: float,
        grade_avg: float,
        n_stops: int,
        temperature_C: float,
        hour: int,
    ) -> float:
        """
        Predict trip energy using ML model.
        Falls back to linear estimate if no model loaded.
        """
        if self.ml_model is not None:
            features = np.array([[
                distance_km, n_passengers, avg_speed_kmh,
                grade_avg, n_stops, temperature_C, hour
            ]])
            return float(self.ml_model.predict(features)[0])
        else:
            # Fallback: fixed rate 0.38 kWh/km (fleet average)
            return distance_km * 0.38

    def _compute_deadhead_energy(
        self, vehicle: VehicleState, lat: float, lon: float,
        temperature_C: float, hour: int
    ) -> tuple:
        """
        Compute energy and time for deadhead (empty driving).
        Returns (distance_km, energy_kWh, travel_time_sec).
        """
        dist = self._haversine_km(vehicle.lat, vehicle.lon, lat, lon)
        if dist < 0.01:
            return 0.0, 0.0, 0.0

        avg_speed = 20.0  # km/h assumed for campus
        energy = self._predict_energy(
            distance_km=dist,
            n_passengers=0,
            avg_speed_kmh=avg_speed,
            grade_avg=0.0,
            n_stops=0,
            temperature_C=temperature_C,
            hour=hour,
        )
        travel_time = (dist / avg_speed) * 3600  # seconds
        return dist, energy, travel_time

    def _compute_trip_energy(self, trip: TripRequest) -> float:
        """Predict energy for the trip itself (pickup → dropoff)."""
        return self._predict_energy(
            distance_km=trip.distance_km,
            n_passengers=trip.n_passengers,
            avg_speed_kmh=trip.avg_speed_kmh,
            grade_avg=trip.grade_avg,
            n_stops=trip.n_stops,
            temperature_C=trip.temperature_C,
            hour=trip.hour,
        )

    # -----------------------------------------------------------------
    # MILP CONSTRUCTION & SOLVING
    # -----------------------------------------------------------------

    def solve(
        self,
        vehicles: list,
        trips: list,
        stations: list,
        current_time: float = 0.0,
        verbose: bool = False,
    ) -> dict:
        """
        Construct and solve the MILP for one decision epoch.

        Parameters
        ----------
        vehicles : list[VehicleState]
            Available (idle) vehicles.
        trips : list[TripRequest]
            Pending trip requests.
        stations : list[StationState]
            Charging stations with availability info.
        current_time : float
            Current simulation time (seconds since midnight).
        verbose : bool
            Print solver output.

        Returns
        -------
        dict with keys:
            'assignments': list of dicts with vehicle_id, trip_id, energy, wait
            'charge_orders': list of dicts with vehicle_id, station info
            'unserved': list of trip_ids not assigned
            'objective': objective value
            'status': solver status string
            'energy_total_kWh': total assigned energy
            'wait_total_sec': total assigned wait time
            'solve_time_sec': wall-clock solver time
            'n_variables': number of decision variables
            'n_constraints': number of constraints
        """
        t_start = _time.time()

        n_v = len(vehicles)
        n_t = len(trips)
        n_s = len(stations)

        # Handle edge cases
        if n_v == 0 or (n_t == 0 and n_v == 0):
            return self._empty_solution(trips, 0.0)

        # --- Pre-compute parameters ---
        if n_t > 0:
            temperature = trips[0].temperature_C
            hour = trips[0].hour
        else:
            temperature = 4.0
            hour = int(current_time / 3600) % 24

        # Energy matrix: e_trip[v][t] = deadhead + trip energy
        # Wait matrix:   w_wait[v][t] = deadhead travel time (sec)
        # Feasibility:   feasible[v][t] = 1 if vehicle can serve trip
        e_trip = np.zeros((n_v, n_t))
        w_wait = np.zeros((n_v, n_t))
        feasible = np.zeros((n_v, n_t), dtype=int)

        cap = self.params.battery_capacity

        for vi, veh in enumerate(vehicles):
            available_energy = (veh.soc - self.soc_min) * cap

            for ti, trip in enumerate(trips):
                # Deadhead: vehicle location → pickup
                dh_dist, dh_energy, dh_time = self._compute_deadhead_energy(
                    veh, trip.origin_lat, trip.origin_lon,
                    temperature, hour
                )
                # Trip: pickup → dropoff
                trip_energy = self._compute_trip_energy(trip)

                total_energy = dh_energy + trip_energy
                e_trip[vi][ti] = total_energy
                w_wait[vi][ti] = dh_time

                # Feasibility check
                if total_energy <= available_energy:
                    feasible[vi][ti] = 1

        # Station parameters
        # e_station[v][s] = energy to reach station
        # reachable[v][s] = 1 if vehicle can reach station
        e_station = np.zeros((n_v, n_s))
        reachable = np.zeros((n_v, n_s), dtype=int)
        needs_charge = np.zeros(n_v, dtype=int)

        for vi, veh in enumerate(vehicles):
            available_energy = (veh.soc - self.soc_min) * cap
            if veh.soc <= self.soc_charge_trigger:
                needs_charge[vi] = 1

            for si, stn in enumerate(stations):
                dist = self._haversine_km(
                    veh.lat, veh.lon, stn.lat, stn.lon
                )
                energy_to_station = self._predict_energy(
                    distance_km=dist,
                    n_passengers=0,
                    avg_speed_kmh=20.0,
                    grade_avg=0.0,
                    n_stops=0,
                    temperature_C=temperature,
                    hour=hour,
                )
                e_station[vi][si] = energy_to_station
                if energy_to_station <= available_energy:
                    reachable[vi][si] = 1

        # --- Build MILP ---
        prob = pulp.LpProblem("Fleet_Dispatch", pulp.LpMinimize)

        # Decision variables
        x = {}  # x[v,t] = vehicle v assigned to trip t
        for vi in range(n_v):
            for ti in range(n_t):
                x[vi, ti] = pulp.LpVariable(
                    f"x_{vi}_{ti}", cat='Binary'
                )

        c = {}  # c[v,s] = vehicle v sent to station s
        for vi in range(n_v):
            for si in range(n_s):
                c[vi, si] = pulp.LpVariable(
                    f"c_{vi}_{si}", cat='Binary'
                )

        # --- Objective ---
        # Energy term
        energy_term = pulp.lpSum(
            self.alpha * e_trip[vi][ti] * x[vi, ti]
            for vi in range(n_v)
            for ti in range(n_t)
        )

        # Wait time term
        wait_term = pulp.lpSum(
            self.beta * w_wait[vi][ti] * x[vi, ti]
            for vi in range(n_v)
            for ti in range(n_t)
        )

        # Unserved penalty term
        penalty_term = pulp.lpSum(
            self.gamma * (1 - pulp.lpSum(x[vi, ti] for vi in range(n_v)))
            for ti in range(n_t)
        )

        prob += energy_term + wait_term + penalty_term, "Total_Objective"

        # --- Constraints ---

        # C1: Each trip assigned to at most one vehicle
        for ti in range(n_t):
            prob += (
                pulp.lpSum(x[vi, ti] for vi in range(n_v)) <= 1,
                f"C1_trip_{ti}"
            )

        # C2: Each vehicle does at most one action
        for vi in range(n_v):
            prob += (
                pulp.lpSum(x[vi, ti] for ti in range(n_t))
                + pulp.lpSum(c[vi, si] for si in range(n_s))
                <= 1,
                f"C2_vehicle_{vi}"
            )

        # C3: Energy feasibility (pre-filtered)
        for vi in range(n_v):
            for ti in range(n_t):
                if feasible[vi][ti] == 0:
                    prob += (
                        x[vi, ti] == 0,
                        f"C3_infeasible_{vi}_{ti}"
                    )

        # C4: Station capacity
        for si in range(n_s):
            prob += (
                pulp.lpSum(c[vi, si] for vi in range(n_v))
                <= stations[si].available_chargers,
                f"C4_station_{si}"
            )

        # C5: Only low-SoC vehicles can charge
        for vi in range(n_v):
            if needs_charge[vi] == 0:
                prob += (
                    pulp.lpSum(c[vi, si] for si in range(n_s)) == 0,
                    f"C5_no_charge_{vi}"
                )

        # C6: Station reachability (pre-filtered)
        for vi in range(n_v):
            for si in range(n_s):
                if reachable[vi][si] == 0:
                    prob += (
                        c[vi, si] == 0,
                        f"C6_unreachable_{vi}_{si}"
                    )

        # --- Solve ---
        # Use Gurobi if available, fall back to CBC
        try:
            solver = pulp.GUROBI(
                timeLimit=self.time_limit_sec,
                gapRel=self.mip_gap,
                msg=verbose,
            )
        except Exception:
            try:
                solver = pulp.GUROBI_CMD(
                    timeLimit=self.time_limit_sec,
                    gapRel=self.mip_gap,
                    msg=verbose,
                )
            except Exception:
                solver = pulp.PULP_CBC_CMD(
                    timeLimit=self.time_limit_sec,
                    gapRel=self.mip_gap,
                    msg=verbose,
                )

        prob.solve(solver)

        solve_time = _time.time() - t_start
        status = pulp.LpStatus[prob.status]

        # --- Extract solution ---
        assignments = []
        charge_orders = []
        served_trips = set()

        for vi in range(n_v):
            for ti in range(n_t):
                if pulp.value(x[vi, ti]) is not None and pulp.value(x[vi, ti]) > 0.5:
                    assignments.append({
                        'vehicle_id': vehicles[vi].vehicle_id,
                        'trip_id': trips[ti].trip_id,
                        'energy_kWh': e_trip[vi][ti],
                        'wait_time_sec': w_wait[vi][ti],
                    })
                    served_trips.add(ti)

            for si in range(n_s):
                if pulp.value(c[vi, si]) is not None and pulp.value(c[vi, si]) > 0.5:
                    charge_orders.append({
                        'vehicle_id': vehicles[vi].vehicle_id,
                        'station_name': stations[si].name,
                        'station_lat': stations[si].lat,
                        'station_lon': stations[si].lon,
                        'energy_to_station': e_station[vi][si],
                    })

        unserved = [
            trips[ti].trip_id
            for ti in range(n_t)
            if ti not in served_trips
        ]

        energy_total = sum(a['energy_kWh'] for a in assignments)
        wait_total = sum(a['wait_time_sec'] for a in assignments)

        solution = {
            'assignments': assignments,
            'charge_orders': charge_orders,
            'unserved': unserved,
            'objective': pulp.value(prob.objective),
            'status': status,
            'energy_total_kWh': round(energy_total, 4),
            'wait_total_sec': round(wait_total, 2),
            'trips_served': len(assignments),
            'trips_unserved': len(unserved),
            'trips_total': n_t,
            'vehicles_charging': len(charge_orders),
            'solve_time_sec': round(solve_time, 4),
            'n_variables': prob.numVariables(),
            'n_constraints': prob.numConstraints(),
        }

        return solution

    def _empty_solution(self, trips, solve_time):
        """Return empty solution when no vehicles or trips."""
        return {
            'assignments': [],
            'charge_orders': [],
            'unserved': [t.trip_id for t in trips],
            'objective': 0.0,
            'status': 'No vehicles',
            'energy_total_kWh': 0.0,
            'wait_total_sec': 0.0,
            'trips_served': 0,
            'trips_unserved': len(trips),
            'trips_total': len(trips),
            'vehicles_charging': 0,
            'solve_time_sec': solve_time,
            'n_variables': 0,
            'n_constraints': 0,
        }


# =============================================================================
# VALIDATION TEST
# =============================================================================

def run_validation_test():
    """
    Validate the MILP with a small hand-crafted scenario.

    Scenario: 3 vehicles, 4 trips, 2 stations
    - V0: high SoC (0.8), near trip T0 and T1
    - V1: low SoC (0.25), needs to charge → should go to station
    - V2: medium SoC (0.5), near trip T2 and T3

    Expected behavior:
    - V0 assigned to T0 or T1 (whichever minimizes energy+wait)
    - V1 sent to charge (SoC < 0.30)
    - V2 assigned to T2 or T3
    - One trip left unserved (3 vehicles, 1 charging → 2 dispatchers, 4 trips)
    """
    print("=" * 60)
    print("MILP VALIDATION TEST")
    print("=" * 60)

    params = VehicleParams()

    vehicles = [
        VehicleState(0, 41.8356, -87.6259, soc=0.80, status='idle'),
        VehicleState(1, 41.8340, -87.6240, soc=0.25, status='idle'),
        VehicleState(2, 41.8350, -87.6270, soc=0.50, status='idle'),
    ]

    trips = [
        TripRequest(
            trip_id=0, origin='A', origin_lat=41.8358, origin_lon=-87.6260,
            destination='B', dest_lat=41.8340, dest_lon=-87.6250,
            distance_km=0.3, n_passengers=4, request_time=36000,
            avg_speed_kmh=20, temperature_C=4.0, grade_avg=0.0,
            n_stops=1, hour=10,
        ),
        TripRequest(
            trip_id=1, origin='C', origin_lat=41.8345, origin_lon=-87.6265,
            destination='D', dest_lat=41.8360, dest_lon=-87.6280,
            distance_km=0.5, n_passengers=6, request_time=36000,
            avg_speed_kmh=20, temperature_C=4.0, grade_avg=0.002,
            n_stops=1, hour=10,
        ),
        TripRequest(
            trip_id=2, origin='E', origin_lat=41.8348, origin_lon=-87.6272,
            destination='F', dest_lat=41.8335, dest_lon=-87.6245,
            distance_km=0.45, n_passengers=3, request_time=36000,
            avg_speed_kmh=20, temperature_C=4.0, grade_avg=-0.001,
            n_stops=0, hour=10,
        ),
        TripRequest(
            trip_id=3, origin='G', origin_lat=41.8370, origin_lon=-87.6290,
            destination='H', dest_lat=41.8340, dest_lon=-87.6250,
            distance_km=0.8, n_passengers=5, request_time=36000,
            avg_speed_kmh=20, temperature_C=4.0, grade_avg=0.003,
            n_stops=2, hour=10,
        ),
    ]

    stations = [
        StationState('Station Alpha', 41.8337, -87.6258, total_chargers=2),
        StationState('Station Beta', 41.8332, -87.6273, total_chargers=2),
    ]

    # Solve WITHOUT ML model (uses fallback linear estimate)
    milp = FleetMILP(
        vehicle_params=params,
        ml_model_path=None,
        alpha=1.0,
        beta=0.01,
        gamma=10.0,
    )

    print("\nSolving (no ML model — fixed energy rate)...")
    sol = milp.solve(vehicles, trips, stations, current_time=36000, verbose=False)

    print(f"\nStatus: {sol['status']}")
    print(f"Objective: {sol['objective']:.4f}")
    print(f"Solve time: {sol['solve_time_sec']:.4f} sec")
    print(f"Variables: {sol['n_variables']}, Constraints: {sol['n_constraints']}")
    print(f"\nTrips served: {sol['trips_served']}/{sol['trips_total']}")
    print(f"Trips unserved: {sol['trips_unserved']}")
    print(f"Vehicles charging: {sol['vehicles_charging']}")
    print(f"Total energy: {sol['energy_total_kWh']:.4f} kWh")
    print(f"Total wait: {sol['wait_total_sec']:.2f} sec")

    print("\nAssignments:")
    for a in sol['assignments']:
        print(f"  Vehicle {a['vehicle_id']} → Trip {a['trip_id']} "
              f"(energy={a['energy_kWh']:.4f} kWh, wait={a['wait_time_sec']:.1f}s)")

    print("\nCharge orders:")
    for co in sol['charge_orders']:
        print(f"  Vehicle {co['vehicle_id']} → {co['station_name']} "
              f"(energy_to_station={co['energy_to_station']:.4f} kWh)")

    # --- Validation checks ---
    print("\n" + "-" * 40)
    print("VALIDATION CHECKS")
    print("-" * 40)

    checks_passed = 0
    total_checks = 7

    # Check 1: Optimal solution found
    if sol['status'] == 'Optimal':
        print("  ✓ Check 1: Optimal solution found")
        checks_passed += 1
    else:
        print(f"  ✗ Check 1: Expected Optimal, got {sol['status']}")

    # Check 2: V1 (low SoC) should be charging
    charging_vehicles = [co['vehicle_id'] for co in sol['charge_orders']]
    if 1 in charging_vehicles:
        print("  ✓ Check 2: V1 (low SoC) sent to charge")
        checks_passed += 1
    else:
        print("  ✗ Check 2: V1 should be charging")

    # Check 3: No more than 1 action per vehicle
    vehicle_actions = {}
    for a in sol['assignments']:
        vehicle_actions[a['vehicle_id']] = vehicle_actions.get(a['vehicle_id'], 0) + 1
    for co in sol['charge_orders']:
        vehicle_actions[co['vehicle_id']] = vehicle_actions.get(co['vehicle_id'], 0) + 1
    max_actions = max(vehicle_actions.values()) if vehicle_actions else 0
    if max_actions <= 1:
        print("  ✓ Check 3: Each vehicle does at most 1 action")
        checks_passed += 1
    else:
        print(f"  ✗ Check 3: Max actions per vehicle = {max_actions}")

    # Check 4: Each trip assigned at most once
    assigned_trips = [a['trip_id'] for a in sol['assignments']]
    if len(assigned_trips) == len(set(assigned_trips)):
        print("  ✓ Check 4: No trip double-assigned")
        checks_passed += 1
    else:
        print("  ✗ Check 4: Duplicate trip assignments")

    # Check 5: 2 trips served (only 2 available dispatchers)
    if sol['trips_served'] == 2:
        print("  ✓ Check 5: 2 trips served (V0 + V2 dispatch, V1 charges)")
        checks_passed += 1
    else:
        print(f"  ✗ Check 5: Expected 2 served, got {sol['trips_served']}")

    # Check 6: 2 trips unserved
    if sol['trips_unserved'] == 2:
        print("  ✓ Check 6: 2 trips unserved (correct)")
        checks_passed += 1
    else:
        print(f"  ✗ Check 6: Expected 2 unserved, got {sol['trips_unserved']}")

    # Check 7: Non-negative energy
    if sol['energy_total_kWh'] >= 0:
        print("  ✓ Check 7: Non-negative total energy")
        checks_passed += 1
    else:
        print(f"  ✗ Check 7: Negative energy {sol['energy_total_kWh']}")

    print(f"\n  Result: {checks_passed}/{total_checks} checks passed")

    return sol


if __name__ == "__main__":
    run_validation_test()