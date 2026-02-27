"""
Rolling Horizon Controller — Real-Time MILP Dispatch
=====================================================

Wraps the MILP solver in a time-stepping simulation that processes
trip requests in 5-minute decision epochs. At each epoch:
    1. Collect trip requests that arrived since last epoch
    2. Complete finished tasks (trips, charging)
    3. Identify idle vehicles
    4. Call FleetMILP.solve() with current state
    5. Execute assignments: update vehicle positions, SoC
    6. Advance clock

Sub-epoch dispatch rounds allow vehicles to serve multiple trips
per epoch window, achieving realistic throughput.

Architecture:
    At each decision epoch (every Δ minutes):
    1. Collect trip requests that arrived since last epoch
    2. Identify idle vehicles (not en-route or charging)
    3. Call FleetMILP.solve() with current state
    4. Execute assignments: update vehicle positions, SoC
    5. Process charging completions
    6. Advance clock

Comparison Baselines:
    - MILP + ML energy prediction (our contribution)
    - MILP + fixed energy rate (ablation — no ML)
    - Nearest-available vehicle (greedy heuristic)

References:
    - Bongiovanni et al. (2019) — e-ADARP rolling horizon
    - Bertsimas et al. (2019) — online vehicle routing with reoptimization

Pipeline Position: Stage 6 (Rolling Horizon Execution)
    trip_data.csv + trained_model.pkl + selected_stations.csv
        → [THIS SCRIPT] → rolling_horizon_results.json,
                           rolling_horizon_trips.csv,
                           rolling_horizon_vehicle_trace.csv,
                           scenario_comparison.json

Inputs:
    - trip_data.csv (from fleet simulator — ~4,186 trips)
    - trained_model.pkl (XGBoost energy predictor)
    - Station locations and vehicle initial states

Outputs:
    - rolling_horizon_results.json (aggregate metrics)
    - rolling_horizon_trips.csv (per-trip dispatch details)
    - rolling_horizon_vehicle_trace.csv (vehicle state over time)
    - scenario_comparison.json (3-strategy comparison)
"""

import numpy as np
import pandas as pd
import json
import time as _time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from copy import deepcopy

# Import MILP solver (lives in 03_MILP)
import sys
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent / "03_MILP"))
from MILP_router import (
    FleetMILP, VehicleState, TripRequest, StationState, VehicleParams
)


# =============================================================================
# SIMULATION VEHICLE (extended state tracking)
# =============================================================================

@dataclass
class SimVehicle:
    """Vehicle with full state tracking for simulation."""
    vehicle_id: int
    lat: float
    lon: float
    soc: float
    status: str = 'idle'           # idle, en_route, serving, charging
    # Current task info
    assigned_trip_id: Optional[int] = None
    assigned_station: Optional[str] = None
    task_end_time: float = 0.0     # when current task completes
    # Destination (for position update)
    dest_lat: float = 0.0
    dest_lon: float = 0.0
    # Daily stats
    km_today: float = 0.0
    energy_today: float = 0.0
    trips_served: int = 0

    def to_milp_state(self) -> VehicleState:
        """Convert to MILP input format."""
        return VehicleState(
            vehicle_id=self.vehicle_id,
            lat=self.lat,
            lon=self.lon,
            soc=self.soc,
            status=self.status,
            km_today=self.km_today,
        )


# =============================================================================
# ROLLING HORIZON CONTROLLER
# =============================================================================

class RollingHorizonController:
    """
    Time-stepping simulation with MILP dispatch.

    Parameters
    ----------
    trip_data : pd.DataFrame
        Trip requests from fleet simulator (trip_data.csv).
    station_info : list of dict
        Station locations: [{'name', 'lat', 'lon', 'chargers'}, ...]
    vehicle_starts : list of dict
        Vehicle initial positions: [{'id', 'lat', 'lon'}, ...]
    ml_model_path : str or None
        Path to trained_model.pkl. None → fixed energy rate.
    epoch_minutes : float
        Decision interval in minutes (default 5).
    n_vehicles : int
        Fleet size (default 10).
    """

    def __init__(
        self,
        trip_data: pd.DataFrame,
        station_info: List[Dict],
        vehicle_starts: List[Dict],
        ml_model_path: Optional[str] = None,
        epoch_minutes: float = 5.0,
        n_vehicles: int = 10,
        # MILP parameters
        alpha: float = 1.0,
        beta: float = 0.01,
        gamma: float = 10.0,
        soc_min: float = 0.10,
        soc_charge_trigger: float = 0.30,
        charge_rate_kw: float = 19.2,
        battery_capacity: float = 118.0,
        campus_speed_kmh: float = 20.0,
    ):
        self.trip_data = trip_data.sort_values('timestamp_sec').reset_index(drop=True)
        self.epoch_sec = epoch_minutes * 60.0
        self.n_vehicles = n_vehicles
        self.campus_speed = campus_speed_kmh
        self.charge_rate = charge_rate_kw
        self.battery_cap = battery_capacity
        self.soc_min = soc_min
        self.soc_charge_trigger = soc_charge_trigger

        # Initialize MILP solver
        self.vparams = VehicleParams(
            battery_capacity=battery_capacity,
            curb_weight=6577.0,
            charge_rate=charge_rate_kw,
        )
        self.milp = FleetMILP(
            vehicle_params=self.vparams,
            ml_model_path=ml_model_path,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            soc_min=soc_min,
            soc_charge_trigger=soc_charge_trigger,
            time_limit_sec=10.0,   # tighter limit for repeated solves
            mip_gap=0.01,
        )

        # Initialize stations
        self.stations = [
            StationState(
                name=s['name'],
                lat=s['lat'],
                lon=s['lon'],
                total_chargers=s.get('chargers', 2),
                occupied_chargers=0,
            )
            for s in station_info
        ]

        # Initialize vehicles
        self.vehicles = []
        starts = vehicle_starts[:n_vehicles]
        for i in range(n_vehicles):
            vs = starts[i % len(starts)]
            self.vehicles.append(SimVehicle(
                vehicle_id=i,
                lat=vs['lat'],
                lon=vs['lon'],
                soc=0.95,  # all start fully charged
            ))

        # Results tracking
        self.dispatch_log = []       # per-trip results
        self.unserved_trips = []     # missed trips
        self.vehicle_trace = []      # vehicle state snapshots
        self.epoch_log = []          # per-epoch summary

    # -----------------------------------------------------------------
    # UTILITY METHODS
    # -----------------------------------------------------------------

    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2) -> float:
        """Great-circle distance in km."""
        R = 6371.0
        rlat1, rlon1 = np.radians(lat1), np.radians(lon1)
        rlat2, rlon2 = np.radians(lat2), np.radians(lon2)
        dlat = rlat2 - rlat1
        dlon = rlon2 - rlon1
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2)
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """Format seconds since midnight as HH:MM."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h:02d}:{m:02d}"

    def _charge_duration(self, current_soc: float, target_soc: float = 0.95) -> float:
        """Estimate charging time in seconds (simplified CC-CV)."""
        energy_needed = (target_soc - current_soc) * self.battery_cap
        if energy_needed <= 0:
            return 0.0
        return (energy_needed / self.charge_rate) * 3600.0

    # -----------------------------------------------------------------
    # STATE UPDATE
    # -----------------------------------------------------------------

    def _complete_tasks(self, current_time: float):
        """Complete any tasks that have finished by current_time."""
        for veh in self.vehicles:
            if veh.status in ('serving', 'en_route') and current_time >= veh.task_end_time:
                # Trip complete — update position to destination
                veh.lat = veh.dest_lat
                veh.lon = veh.dest_lon
                veh.status = 'idle'
                veh.assigned_trip_id = None

            elif veh.status == 'charging' and current_time >= veh.task_end_time:
                # Charging complete — restore SoC
                veh.soc = 0.95
                veh.lat = veh.dest_lat
                veh.lon = veh.dest_lon
                veh.status = 'idle'
                veh.assigned_station = None

                # Free up charger
                for stn in self.stations:
                    if stn.name == veh.assigned_station or (
                        abs(stn.lat - veh.lat) < 0.0001 and
                        abs(stn.lon - veh.lon) < 0.0001
                    ):
                        stn.occupied_chargers = max(0, stn.occupied_chargers - 1)
                        break

    def _log_vehicle_states(self, current_time: float):
        """Snapshot all vehicle states."""
        for v in self.vehicles:
            self.vehicle_trace.append({
                'time_sec': current_time,
                'time_str': self._fmt_time(current_time),
                'vehicle_id': v.vehicle_id,
                'status': v.status,
                'lat': round(v.lat, 6),
                'lon': round(v.lon, 6),
                'soc': round(v.soc, 4),
                'km_today': round(v.km_today, 2),
                'trips_served': v.trips_served,
            })

    # -----------------------------------------------------------------
    # EXECUTE MILP SOLUTION
    # -----------------------------------------------------------------

    def _execute_assignments(
        self,
        solution: dict,
        trips_this_epoch: List[TripRequest],
        current_time: float,
    ):
        """
        Execute MILP solution: assign vehicles to trips and stations.
        Update vehicle states and log results.
        """
        trip_lookup = {t.trip_id: t for t in trips_this_epoch}

        # Process trip assignments
        for assignment in solution['assignments']:
            vid = assignment['vehicle_id']
            tid = assignment['trip_id']
            trip = trip_lookup.get(tid)
            if trip is None:
                continue
            veh = self.vehicles[vid]

            # Calculate travel times
            deadhead_km = self._haversine_km(
                veh.lat, veh.lon, trip.origin_lat, trip.origin_lon
            )
            deadhead_time = (deadhead_km / self.campus_speed) * 3600.0
            trip_time = (trip.distance_km / self.campus_speed) * 3600.0
            total_time = deadhead_time + trip_time

            # Energy consumed (from MILP solution)
            energy_kwh = assignment['energy_kWh']

            # Update vehicle state
            veh.status = 'serving'
            veh.assigned_trip_id = tid
            veh.dest_lat = trip.dest_lat
            veh.dest_lon = trip.dest_lon
            veh.task_end_time = current_time + total_time
            veh.soc -= energy_kwh / self.battery_cap
            veh.soc = max(self.soc_min, veh.soc)
            veh.km_today += deadhead_km + trip.distance_km
            veh.energy_today += energy_kwh
            veh.trips_served += 1

            # Log
            wait_time = current_time - trip.request_time
            self.dispatch_log.append({
                'trip_id': tid,
                'vehicle_id': vid,
                'request_time': trip.request_time,
                'dispatch_time': current_time,
                'wait_time_sec': max(0, wait_time),
                'deadhead_km': round(deadhead_km, 4),
                'trip_km': trip.distance_km,
                'energy_kWh': round(energy_kwh, 4),
                'soc_after': round(veh.soc, 4),
                'origin': trip.origin,
                'destination': trip.destination,
                'n_passengers': trip.n_passengers,
                'hour': trip.hour,
                'status': 'served',
            })

        # Process charging assignments
        for charge_order in solution['charge_orders']:
            vid = charge_order['vehicle_id']
            veh = self.vehicles[vid]
            stn_name = charge_order['station_name']

            # Travel to station
            station_dist = self._haversine_km(
                veh.lat, veh.lon,
                charge_order['station_lat'], charge_order['station_lon']
            )
            travel_time = (station_dist / self.campus_speed) * 3600.0
            charge_time = self._charge_duration(veh.soc)

            # Update vehicle
            veh.status = 'charging'
            veh.assigned_station = stn_name
            veh.dest_lat = charge_order['station_lat']
            veh.dest_lon = charge_order['station_lon']
            veh.task_end_time = current_time + travel_time + charge_time

            # Energy to reach station
            energy_to_station = charge_order['energy_to_station']
            veh.soc -= energy_to_station / self.battery_cap
            veh.soc = max(self.soc_min, veh.soc)
            veh.km_today += station_dist

            # Occupy charger
            for stn in self.stations:
                if stn.name == stn_name:
                    stn.occupied_chargers = min(
                        stn.occupied_chargers + 1, stn.total_chargers
                    )

        # Unserved trips stay in pending queue for re-attempt next epoch.

    # -----------------------------------------------------------------
    # GREEDY BASELINE (Nearest-Available)
    # -----------------------------------------------------------------

    def _greedy_dispatch(
        self, pending_trips: List[TripRequest], current_time: float
    ) -> List[int]:
        """
        Nearest-available vehicle dispatch (no MILP).
        Returns list of served trip IDs.
        """
        served_ids = []

        for trip in pending_trips:
            # Find nearest idle vehicle with enough SoC
            best_vehicle = None
            best_dist = float('inf')

            for veh in self.vehicles:
                if veh.status != 'idle':
                    continue
                if veh.soc <= self.soc_charge_trigger:
                    continue

                dist = self._haversine_km(
                    veh.lat, veh.lon, trip.origin_lat, trip.origin_lon
                )
                total_dist = dist + trip.distance_km
                energy_est = total_dist * 0.38  # fixed rate for baseline
                available = (veh.soc - self.soc_min) * self.battery_cap

                if energy_est <= available and dist < best_dist:
                    best_dist = dist
                    best_vehicle = veh

            if best_vehicle is None:
                continue

            # Assign
            veh = best_vehicle
            deadhead_km = best_dist
            deadhead_time = (deadhead_km / self.campus_speed) * 3600.0
            trip_time = (trip.distance_km / self.campus_speed) * 3600.0
            total_time = deadhead_time + trip_time

            energy_kwh = (deadhead_km + trip.distance_km) * 0.38

            veh.status = 'serving'
            veh.assigned_trip_id = trip.trip_id
            veh.dest_lat = trip.dest_lat
            veh.dest_lon = trip.dest_lon
            veh.task_end_time = current_time + total_time
            veh.soc -= energy_kwh / self.battery_cap
            veh.soc = max(self.soc_min, veh.soc)
            veh.km_today += deadhead_km + trip.distance_km
            veh.energy_today += energy_kwh
            veh.trips_served += 1

            wait_time = current_time - trip.request_time
            self.dispatch_log.append({
                'trip_id': trip.trip_id,
                'vehicle_id': veh.vehicle_id,
                'request_time': trip.request_time,
                'dispatch_time': current_time,
                'wait_time_sec': max(0, wait_time),
                'deadhead_km': round(deadhead_km, 4),
                'trip_km': trip.distance_km,
                'energy_kWh': round(energy_kwh, 4),
                'soc_after': round(veh.soc, 4),
                'origin': trip.origin,
                'destination': trip.destination,
                'n_passengers': trip.n_passengers,
                'hour': trip.hour,
                'status': 'served',
            })
            served_ids.append(trip.trip_id)

        # Send low-SoC vehicles to nearest station
        for veh in self.vehicles:
            if veh.status == 'idle' and veh.soc <= self.soc_charge_trigger:
                best_stn = None
                best_dist = float('inf')
                for stn in self.stations:
                    if stn.occupied_chargers >= stn.total_chargers:
                        continue
                    d = self._haversine_km(veh.lat, veh.lon, stn.lat, stn.lon)
                    if d < best_dist:
                        best_dist = d
                        best_stn = stn

                if best_stn:
                    travel_time = (best_dist / self.campus_speed) * 3600.0
                    charge_time = self._charge_duration(veh.soc)
                    veh.status = 'charging'
                    veh.assigned_station = best_stn.name
                    veh.dest_lat = best_stn.lat
                    veh.dest_lon = best_stn.lon
                    veh.task_end_time = current_time + travel_time + charge_time
                    energy_to = best_dist * 0.38
                    veh.soc -= energy_to / self.battery_cap
                    veh.soc = max(self.soc_min, veh.soc)
                    veh.km_today += best_dist
                    best_stn.occupied_chargers += 1

        return served_ids

    # -----------------------------------------------------------------
    # MAIN SIMULATION LOOP
    # -----------------------------------------------------------------

    def run(
        self,
        strategy: str = 'milp',
        verbose: bool = True,
        on_epoch_end=None,
        max_epochs: Optional[int] = None,
    ) -> dict:
        """
        Execute the rolling horizon simulation for the full day.

        Parameters
        ----------
        strategy : str
            'milp' for MILP dispatch, 'greedy' for nearest-available.
        verbose : bool
            Print progress.
        on_epoch_end : callable, optional
            If set, called after each epoch with
            (epoch_idx, current_time, vehicles_snapshot, n_served, n_pending).
            vehicles_snapshot is a list of dicts with keys vehicle_id, lat, lon, status, soc.
            Used for real-time visualization.
        max_epochs : int, optional
            If set, stop after this many epochs (for testing or real-time demos).

        Returns
        -------
        dict : Aggregate results and metrics.
        """
        t_wall_start = _time.time()

        # Time bounds from trip data
        sim_start = self.trip_data['timestamp_sec'].min() - 60
        sim_end = self.trip_data['timestamp_sec'].max() + 300
        n_epochs = int(np.ceil((sim_end - sim_start) / self.epoch_sec))
        if max_epochs is not None:
            n_epochs = min(n_epochs, max_epochs)

        if verbose:
            print("=" * 70)
            print("ROLLING HORIZON SIMULATION")
            print("=" * 70)
            print(f"  Strategy: {strategy.upper()}")
            print(f"  Time range: {self._fmt_time(sim_start)} — {self._fmt_time(sim_end)}")
            print(f"  Epoch interval: {self.epoch_sec/60:.0f} min")
            print(f"  Total epochs: {n_epochs}")
            print(f"  Vehicles: {self.n_vehicles}")
            print(f"  Stations: {len(self.stations)}")
            print(f"  Total trips: {len(self.trip_data)}")
            ml_status = 'loaded' if self.milp.ml_model else 'none (fixed rate)'
            print(f"  ML model: {ml_status}")
            print()

        trip_pointer = 0
        pending_trips = []
        total_milp_time = 0.0
        epochs_solved = 0

        for epoch_idx in range(n_epochs):
            current_time = sim_start + epoch_idx * self.epoch_sec
            epoch_end = current_time + self.epoch_sec

            # --- Step 1: Collect new trip requests in this window ---
            new_trips = []
            while (trip_pointer < len(self.trip_data) and
                   self.trip_data.iloc[trip_pointer]['timestamp_sec'] < epoch_end):
                row = self.trip_data.iloc[trip_pointer]
                new_trips.append(TripRequest(
                    trip_id=int(row['trip_id']),
                    origin=row['origin'],
                    origin_lat=row['origin_lat'],
                    origin_lon=row['origin_lon'],
                    destination=row['destination'],
                    dest_lat=row['dest_lat'],
                    dest_lon=row['dest_lon'],
                    distance_km=row['distance_km'],
                    n_passengers=int(row['n_passengers']),
                    request_time=row['timestamp_sec'],
                    avg_speed_kmh=row.get('avg_speed_kmh', 20.0),
                    temperature_C=row.get('temperature_C', 4.0),
                    grade_avg=row.get('grade_avg', 0.0),
                    n_stops=int(row.get('n_stops', 1)),
                    hour=int(row.get('hour', 12)),
                ))
                trip_pointer += 1

            pending_trips.extend(new_trips)

            # Optional callback for real-time viz (called every epoch)
            if on_epoch_end is not None:
                vehicle_snapshot = [
                    {'vehicle_id': v.vehicle_id, 'lat': v.lat, 'lon': v.lon, 'status': v.status, 'soc': v.soc}
                    for v in self.vehicles
                ]
                on_epoch_end(
                    epoch_idx, current_time, vehicle_snapshot,
                    len(self.dispatch_log), len(pending_trips),
                )

            # Skip epoch if no pending trips
            if not pending_trips:
                continue

            # --- Step 2: Sub-epoch dispatch rounds ---
            sub_steps = 6
            sub_interval = self.epoch_sec / sub_steps

            for sub in range(sub_steps):
                sub_time = current_time + sub * sub_interval

                # Complete finished tasks
                self._complete_tasks(sub_time)

                if not pending_trips:
                    break

                if strategy == 'milp':
                    # Get idle vehicles
                    idle_vehicles = [
                        v.to_milp_state()
                        for v in self.vehicles
                        if v.status == 'idle'
                    ]

                    if not idle_vehicles:
                        continue

                    # Solve MILP
                    sol = self.milp.solve(
                        vehicles=idle_vehicles,
                        trips=pending_trips,
                        stations=self.stations,
                        current_time=sub_time,
                        verbose=False,
                    )

                    total_milp_time += sol['solve_time_sec']
                    epochs_solved += 1

                    # Execute solution
                    self._execute_assignments(sol, pending_trips, sub_time)

                    # Remove served trips from pending
                    served_ids = {a['trip_id'] for a in sol['assignments']}
                    pending_trips = [t for t in pending_trips if t.trip_id not in served_ids]

                elif strategy == 'greedy':
                    served_ids = self._greedy_dispatch(pending_trips, sub_time)
                    pending_trips = [t for t in pending_trips if t.trip_id not in served_ids]

            # --- Step 3: Timeout old pending trips ---
            max_wait = 30 * 60  # 30 minutes
            still_pending = []
            for trip in pending_trips:
                if epoch_end - trip.request_time > max_wait:
                    self.unserved_trips.append({
                        'trip_id': trip.trip_id,
                        'request_time': trip.request_time,
                        'origin': trip.origin,
                        'destination': trip.destination,
                        'n_passengers': trip.n_passengers,
                        'hour': trip.hour,
                        'reason': 'timeout_30min',
                    })
                else:
                    still_pending.append(trip)
            pending_trips = still_pending

            # Log vehicle states periodically
            if epoch_idx % 12 == 0:  # every hour
                self._log_vehicle_states(current_time)

            # Progress
            if verbose and epoch_idx % 24 == 0:
                n_served = len(self.dispatch_log)
                print(f"  Epoch {epoch_idx:4d}/{n_epochs} "
                      f"({self._fmt_time(current_time)}) — "
                      f"served: {n_served}, pending: {len(pending_trips)}")

        # Mark remaining pending as unserved
        for trip in pending_trips:
            self.unserved_trips.append({
                'trip_id': trip.trip_id,
                'request_time': trip.request_time,
                'origin': trip.origin,
                'destination': trip.destination,
                'n_passengers': trip.n_passengers,
                'hour': trip.hour,
                'reason': 'end_of_day',
            })

        # Final vehicle state log
        self._log_vehicle_states(sim_end)

        # --- Compile results ---
        wall_time = _time.time() - t_wall_start
        results = self._compile_results(
            strategy, total_milp_time, epochs_solved, wall_time
        )

        if verbose:
            print(f"\n{'=' * 70}")
            print("SIMULATION COMPLETE")
            print(f"{'=' * 70}")
            print(f"  Trips served: {results['trips_served']}/{results['total_trips']} "
                  f"({results['service_rate_pct']:.1f}%)")
            print(f"  Trips missed: {results['trips_unserved']}")
            print(f"  Total energy: {results['total_energy_kWh']:.1f} kWh")
            print(f"  Avg wait: {results['avg_wait_sec']:.1f} sec")
            print(f"  Total deadhead: {results['total_deadhead_km']:.1f} km")
            print(f"  Wall time: {wall_time:.1f} sec")
            if epochs_solved > 0:
                print(f"  MILP solves: {epochs_solved}, "
                      f"avg time: {total_milp_time/epochs_solved:.4f} sec")

        return results

    def _compile_results(
        self, strategy, milp_time, epochs_solved, wall_time
    ) -> dict:
        """Aggregate simulation results."""
        n_served = len(self.dispatch_log)
        n_unserved = len(self.unserved_trips)
        total_trips = n_served + n_unserved

        total_energy = sum(d['energy_kWh'] for d in self.dispatch_log)
        total_deadhead = sum(d['deadhead_km'] for d in self.dispatch_log)
        total_trip_km = sum(d['trip_km'] for d in self.dispatch_log)
        avg_wait = (
            np.mean([d['wait_time_sec'] for d in self.dispatch_log])
            if self.dispatch_log else 0.0
        )

        return {
            'strategy': strategy,
            'total_trips': total_trips,
            'trips_served': n_served,
            'trips_unserved': n_unserved,
            'service_rate_pct': round(n_served / max(total_trips, 1) * 100, 2),
            'total_energy_kWh': round(total_energy, 2),
            'avg_energy_per_trip': round(total_energy / max(n_served, 1), 4),
            'total_deadhead_km': round(total_deadhead, 2),
            'total_trip_km': round(total_trip_km, 2),
            'avg_wait_sec': round(avg_wait, 1),
            'milp_total_time_sec': round(milp_time, 2),
            'milp_solves': epochs_solved,
            'milp_avg_time_sec': round(milp_time / max(epochs_solved, 1), 4),
            'wall_time_sec': round(wall_time, 2),
            'n_vehicles': self.n_vehicles,
            'vehicle_stats': [
                {
                    'vehicle_id': v.vehicle_id,
                    'trips_served': v.trips_served,
                    'km_today': round(v.km_today, 2),
                    'energy_today': round(v.energy_today, 2),
                    'final_soc': round(v.soc, 4),
                }
                for v in self.vehicles
            ],
        }

    def save_results(self, output_dir: str, results: dict):
        """Save all output files."""
        out = Path(output_dir)
        prefix = results.get('strategy', 'milp')

        # Per-trip dispatch log
        if self.dispatch_log:
            df_trips = pd.DataFrame(self.dispatch_log)
            df_trips.to_csv(out / f'{prefix}_trips.csv', index=False)

        # Vehicle trace
        if self.vehicle_trace:
            df_trace = pd.DataFrame(self.vehicle_trace)
            df_trace.to_csv(out / f'{prefix}_vehicle_trace.csv', index=False)

        # Results JSON
        with open(out / f'{prefix}_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    def reset(self, vehicle_starts: List[Dict]):
        """Reset controller state for a new scenario run."""
        self.dispatch_log = []
        self.unserved_trips = []
        self.vehicle_trace = []
        self.epoch_log = []

        # Reset vehicles
        self.vehicles = []
        starts = vehicle_starts[:self.n_vehicles]
        for i in range(self.n_vehicles):
            vs = starts[i % len(starts)]
            self.vehicles.append(SimVehicle(
                vehicle_id=i,
                lat=vs['lat'],
                lon=vs['lon'],
                soc=0.95,
            ))

        # Reset stations
        for stn in self.stations:
            stn.occupied_chargers = 0


# =============================================================================
# MAIN — Run 3-strategy comparison
# =============================================================================

def main(data_dir: str = '.', output_dir: str = None):
    """
    Run rolling horizon with MILP+ML, MILP-Fixed, and Nearest-Available.
    """
    base_dir = Path(data_dir)
    out_dir = Path(output_dir) if output_dir else base_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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
            + "\nPass data_dir= to main() or place trip_data.csv in 01_Data_gen/Outputs/."
        )
    trip_data = pd.read_csv(_trip_path)
    print(f"Loaded {len(trip_data)} trips from {_trip_path}")

    # Station info
    station_info = [
        {'name': 'MTCC', 'lat': 41.838385, 'lon': -87.627555, 'chargers': 2},
        {'name': 'Paul Galvin Library', 'lat': 41.833675, 'lon': -87.628336, 'chargers': 2},
        {'name': 'McCormick Student Village', 'lat': 41.835527, 'lon': -87.624207, 'chargers': 2},
        {'name': 'Crown Hall', 'lat': 41.833199, 'lon': -87.627273, 'chargers': 2},
        {'name': 'Kaplan Institute', 'lat': 41.836861, 'lon': -87.628300, 'chargers': 2},
        {'name': 'Arthur S. Keating Sports Center', 'lat': 41.838985, 'lon': -87.625566, 'chargers': 2},
    ]

    # Vehicle starting positions
    vehicle_starts = [
        {'id': 0, 'lat': 41.837866, 'lon': -87.624703},  # Kacek Hall
        {'id': 1, 'lat': 41.831394, 'lon': -87.627231},  # Michael Galvin Tower
        {'id': 2, 'lat': 41.833199, 'lon': -87.627273},  # S.R. Crown Hall
        {'id': 3, 'lat': 41.835681, 'lon': -87.628387},  # Herman Hall
        {'id': 4, 'lat': 41.836861, 'lon': -87.628300},  # Kaplan Institute
        {'id': 5, 'lat': 41.835527, 'lon': -87.624207},  # McCormick Student Village
        {'id': 6, 'lat': 41.833675, 'lon': -87.628336},  # Paul Galvin Library
        {'id': 7, 'lat': 41.834344, 'lon': -87.623795},  # Farr Hall
        {'id': 8, 'lat': 41.834344, 'lon': -87.623795},  # Farr Hall (duplicate)
        {'id': 9, 'lat': 41.837866, 'lon': -87.624703},  # Kacek Hall (duplicate)
    ]

    ml_model_path = str(base_dir / 'trained_model.pkl')

    # =========================================================================
    # SCENARIO 1: MILP + ML Energy Prediction (our contribution)
    # =========================================================================
    print("\n" + "=" * 70)
    print("SCENARIO 1: MILP + ML Energy Prediction")
    print("=" * 70)

    rh_ml = RollingHorizonController(
        trip_data=trip_data,
        station_info=station_info,
        vehicle_starts=vehicle_starts,
        ml_model_path=ml_model_path,
        epoch_minutes=5.0,
        n_vehicles=10,
    )
    results_ml = rh_ml.run(strategy='milp', verbose=True)
    results_ml['method'] = 'milp_ml'
    rh_ml.save_results(str(out_dir), results_ml)

    # =========================================================================
    # SCENARIO 2: MILP + Fixed Energy Rate (ablation — no ML)
    # =========================================================================
    print("\n" + "=" * 70)
    print("SCENARIO 2: MILP + Fixed Energy Rate (no ML)")
    print("=" * 70)

    rh_fixed = RollingHorizonController(
        trip_data=trip_data,
        station_info=station_info,
        vehicle_starts=vehicle_starts,
        ml_model_path=None,  # No ML → fixed 0.38 kWh/km
        epoch_minutes=5.0,
        n_vehicles=10,
    )
    results_fixed = rh_fixed.run(strategy='milp', verbose=True)
    results_fixed['method'] = 'milp_fixed'
    rh_fixed.save_results(str(out_dir), results_fixed)

    # =========================================================================
    # SCENARIO 3: Nearest-Available (greedy baseline)
    # =========================================================================
    print("\n" + "=" * 70)
    print("SCENARIO 3: Nearest-Available (Greedy Baseline)")
    print("=" * 70)

    rh_greedy = RollingHorizonController(
        trip_data=trip_data,
        station_info=station_info,
        vehicle_starts=vehicle_starts,
        ml_model_path=None,
        epoch_minutes=5.0,
        n_vehicles=10,
    )
    results_baseline = rh_greedy.run(strategy='greedy', verbose=True)
    results_baseline['method'] = 'nearest_available'
    rh_greedy.save_results(str(out_dir), results_baseline)

    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    print("\n" + "=" * 70)
    print("SCENARIO COMPARISON")
    print("=" * 70)

    methods = [results_ml, results_fixed, results_baseline]
    headers = ['MILP+ML', 'MILP-Fixed', 'Nearest']
    print(f"\n  {'Metric':<30} {headers[0]:>12} {headers[1]:>12} {headers[2]:>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")

    metrics = [
        ('Trips Served', 'trips_served'),
        ('Trips Missed', 'trips_unserved'),
        ('Service Rate (%)', 'service_rate_pct'),
        ('Total Energy (kWh)', 'total_energy_kWh'),
        ('Avg Energy/Trip (kWh)', 'avg_energy_per_trip'),
        ('Total Deadhead (km)', 'total_deadhead_km'),
        ('Total Trip km', 'total_trip_km'),
        ('Avg Wait (sec)', 'avg_wait_sec'),
        ('MILP Solves', 'milp_solves'),
        ('MILP Time (sec)', 'milp_total_time_sec'),
        ('Wall Time (sec)', 'wall_time_sec'),
    ]

    for label, key in metrics:
        vals = [str(m.get(key, 'N/A')) for m in methods]
        print(f"  {label:<30} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    # Improvement calculations
    if results_baseline['total_energy_kWh'] > 0:
        energy_improvement = (
            (results_baseline['total_energy_kWh'] - results_ml['total_energy_kWh'])
            / results_baseline['total_energy_kWh'] * 100
        )
        print(f"\n  Energy Reduction (MILP+ML vs Nearest): {energy_improvement:+.1f}%")

    if results_fixed['total_energy_kWh'] > 0:
        ml_vs_fixed = (
            (results_fixed['total_energy_kWh'] - results_ml['total_energy_kWh'])
            / results_fixed['total_energy_kWh'] * 100
        )
        print(f"  Energy Reduction (ML vs Fixed):         {ml_vs_fixed:+.1f}%")

    # Save comparison
    comparison = {
        'milp_ml': results_ml,
        'milp_fixed': results_fixed,
        'nearest_available': results_baseline,
    }
    with open(out_dir / 'scenario_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\n  Results saved to {out_dir}/")
    print("  Files: *_results.json, *_trips.csv, *_vehicle_trace.csv,")
    print("         scenario_comparison.json")


if __name__ == "__main__":
    main()