import numpy as np
import pandas as pd
import heapq
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from enum import Enum


# =============================================================================
# CONSTANTS AND ENUMS
# =============================================================================

class VehicleStatus(Enum):
    IDLE = "idle"
    EN_ROUTE_PICKUP = "en_route_pickup"
    EN_ROUTE_DROPOFF = "en_route_dropoff"
    EN_ROUTE_STATION = "en_route_station"
    CHARGING = "charging"
    WAITING_CHARGER = "waiting_charger"


class EventType(Enum):
    TRIP_REQUEST = "trip_request"
    VEHICLE_AT_PICKUP = "vehicle_at_pickup"
    VEHICLE_AT_DROPOFF = "vehicle_at_dropoff"
    VEHICLE_AT_STATION = "vehicle_at_station"
    CHARGING_COMPLETE = "charging_complete"
    CHARGER_AVAILABLE = "charger_available"


# =============================================================================
# VEHICLE PARAMETERS (GreenPower EV Star MY20)
# =============================================================================

@dataclass
class VehicleParams:
    """GreenPower EV Star parameters."""
    mass_curb: float = 4500.0
    mass_passenger: float = 70.0
    mass_rotational_factor: float = 1.05
    Cd: float = 0.65
    frontal_area: float = 4.65
    Crr: float = 0.010
    eta_drivetrain: float = 0.85
    eta_regen: float = 0.65
    eta_battery: float = 0.95
    battery_capacity: float = 118.0     # kWh
    soc_min: float = 0.15
    soc_max: float = 0.95
    aux_power: float = 2.0             # kW base (before HVAC)
    air_density: float = 1.225
    gravity: float = 9.81

    @property
    def CdA(self) -> float:
        return self.Cd * self.frontal_area

    @property
    def usable_capacity(self) -> float:
        return self.battery_capacity * (self.soc_max - self.soc_min)


# =============================================================================
# ENERGY MODEL (with HVAC temperature dependence)
# =============================================================================

class EnergyModel:
    """
    Physics-based energy model with temperature-dependent HVAC.

    Extends Station_data.py's model by adding HVAC power draw
    that varies with ambient temperature.
    """

    def __init__(self, params: VehicleParams = None):
        self.params = params or VehicleParams()

    def _hvac_power(self, temperature_C: float) -> float:
        """
        HVAC power draw based on ambient temperature.

        Comfort zone: 20-24°C → minimal HVAC
        Below 20°C → heating (up to 6 kW at -10°C)
        Above 24°C → cooling (up to 4 kW at 38°C)
        """
        if 20 <= temperature_C <= 24:
            return 0.3  # Fan only
        elif temperature_C < 20:
            # Heating: linear ramp from 0.3 kW at 20°C to 6 kW at -10°C
            return 0.3 + (20 - temperature_C) * (5.7 / 30)
        else:
            # Cooling: linear ramp from 0.3 kW at 24°C to 4 kW at 38°C
            return 0.3 + (temperature_C - 24) * (3.7 / 14)

    def compute_trip_energy(
        self,
        distance_km: float,
        avg_speed_kmh: float = 20.0,
        n_passengers: int = 5,
        n_stops: int = 0,
        grade_avg: float = 0.0,
        temperature_C: float = 10.0
    ) -> float:
        """
        Compute energy consumption including HVAC.

        Returns energy in kWh.
        """
        p = self.params

        total_mass = p.mass_curb + n_passengers * p.mass_passenger
        effective_mass = total_mass * p.mass_rotational_factor

        distance_m = distance_km * 1000
        speed_ms = avg_speed_kmh / 3.6

        if distance_m <= 0 or speed_ms <= 0:
            return 0.0

        # Force components
        F_rolling = p.Crr * total_mass * p.gravity
        F_aero = 0.5 * p.air_density * p.CdA * speed_ms**2
        F_grade = total_mass * p.gravity * grade_avg

        # Energy components
        E_traction = (F_rolling + F_aero + F_grade) * distance_m

        n_accel_events = max(1, n_stops + 1)
        E_accel_per_stop = 0.5 * effective_mass * speed_ms**2
        E_accel = n_accel_events * E_accel_per_stop * (1 - p.eta_regen)

        E_mech = E_traction + E_accel

        # Convert to electrical
        E_electrical_j = E_mech / (p.eta_drivetrain * p.eta_battery)
        E_electrical_kwh = E_electrical_j / 3_600_000

        # Auxiliary + HVAC
        trip_duration_h = distance_km / avg_speed_kmh
        hvac_power = self._hvac_power(temperature_C)
        E_aux_kwh = (p.aux_power + hvac_power) * trip_duration_h

        return max(0.0, E_electrical_kwh + E_aux_kwh)


# =============================================================================
# DISTANCE UTILITIES
# =============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def road_distance(straight_line_km, circuity=1.3):
    """Approximate road distance from straight-line."""
    return straight_line_km * circuity


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Event:
    """A simulation event, ordered by time."""
    time: float
    event_type: EventType
    data: dict = field(default_factory=dict)

    def __lt__(self, other):
        return self.time < other.time


@dataclass
class Vehicle:
    """State of a single vehicle."""
    vehicle_id: int
    lat: float
    lon: float
    location_name: str
    soc: float
    status: VehicleStatus = VehicleStatus.IDLE
    passengers: int = 0
    current_trip_id: int = None
    km_today: float = 0.0
    trips_served: int = 0
    energy_consumed_kWh: float = 0.0


@dataclass
class ChargingStation:
    """State of a charging station."""
    name: str
    lat: float
    lon: float
    n_chargers: int = 2
    charging_rate_kW: float = 19.2      # Level 2 AC
    vehicles_charging: List[int] = field(default_factory=list)
    queue: List[int] = field(default_factory=list)

    @property
    def has_available_charger(self) -> bool:
        return len(self.vehicles_charging) < self.n_chargers


@dataclass
class TripRequest:
    """An on-demand trip request."""
    trip_id: int
    request_time: float
    origin_name: str
    origin_lat: float
    origin_lon: float
    origin_category: str
    dest_name: str
    dest_lat: float
    dest_lon: float
    dest_category: str
    n_passengers: int
    assigned_vehicle: int = None
    pickup_time: float = None
    dropoff_time: float = None
    wait_time_sec: float = None
    status: str = "pending"


# =============================================================================
# DEMAND GENERATOR (Inhomogeneous Poisson Process)
# =============================================================================

class DemandGenerator:
    """
    Generate individual trip requests using an inhomogeneous Poisson process.

    For each OD pair and each hour, the arrival rate is:
        λ = base_rate × time_multiplier × priority_weight

    Individual arrivals are generated by sampling Poisson counts
    then uniformly distributing within each hour.
    """

    def __init__(
        self,
        pois: pd.DataFrame,
        base_trips_per_hour: float = 8.0,
        operating_hours: Tuple[int, int] = (7, 23),
        seed: int = 42
    ):
        self.pois = pois
        self.base_rate = base_trips_per_hour
        self.start_hour, self.end_hour = operating_hours
        self.rng = np.random.RandomState(seed)

    def get_multiplier(self, hour: int, origin_cat: str, dest_cat: str) -> float:
        """Time-of-day demand multiplier by OD category."""
        if origin_cat == 'Residence' and dest_cat == 'Academic':
            if 7 <= hour <= 9: return 1.0
            elif 10 <= hour <= 11: return 0.3
            else: return 0.1
        elif origin_cat == 'Academic' and dest_cat == 'Residence':
            if 16 <= hour <= 18: return 1.0
            elif 12 <= hour <= 14: return 0.4
            elif 19 <= hour <= 21: return 0.3
            else: return 0.1
        elif dest_cat == 'Dining':
            if 11 <= hour <= 13 or 17 <= hour <= 19: return 1.0
            elif 7 <= hour <= 9: return 0.5
            else: return 0.2
        elif dest_cat == 'Library':
            if 10 <= hour <= 22: return 0.6
            else: return 0.1
        elif dest_cat == 'Recreation':
            if 15 <= hour <= 21: return 0.7
            else: return 0.2
        elif dest_cat == 'Student Center':
            if 9 <= hour <= 20: return 0.7
            else: return 0.2
        else:
            if 8 <= hour <= 18: return 0.5
            else: return 0.2

    def generate_requests(self) -> List[TripRequest]:
        """Generate all trip requests for the operating day."""
        requests = []
        trip_id = 0

        for hour in range(self.start_hour, self.end_hour + 1):
            hour_start_sec = hour * 3600

            for _, origin in self.pois.iterrows():
                for _, dest in self.pois.iterrows():
                    if origin['poi_name'] == dest['poi_name']:
                        continue

                    multiplier = self.get_multiplier(
                        hour, origin['category'], dest['category']
                    )
                    priority_weight = (
                        (5 - origin['priority']) * (5 - dest['priority']) / 16
                    )
                    lam = self.base_rate * multiplier * priority_weight

                    if lam < 0.01:
                        continue

                    n_trips = self.rng.poisson(lam)

                    if n_trips > 0:
                        arrival_offsets = sorted(self.rng.uniform(0, 3600, n_trips))

                        for offset in arrival_offsets:
                            n_pax = max(1, min(12, self.rng.poisson(5)))
                            requests.append(TripRequest(
                                trip_id=trip_id,
                                request_time=hour_start_sec + offset,
                                origin_name=origin['poi_name'],
                                origin_lat=origin['latitude'],
                                origin_lon=origin['longitude'],
                                origin_category=origin['category'],
                                dest_name=dest['poi_name'],
                                dest_lat=dest['latitude'],
                                dest_lon=dest['longitude'],
                                dest_category=dest['category'],
                                n_passengers=n_pax
                            ))
                            trip_id += 1

        requests.sort(key=lambda r: r.request_time)
        return requests


# =============================================================================
# WEATHER PROFILE (Chicago monthly temperatures)
# =============================================================================

def get_temperature_profile(month: int = 3, seed: int = 42) -> callable:
    """
    Return a function that gives temperature at any hour.

    Uses a sinusoidal daily pattern around the monthly average.
    Chicago monthly averages (°C):
    Jan:-3.5, Feb:-1.7, Mar:4.4, Apr:10.6, May:16.7, Jun:22.2,
    Jul:25.0, Aug:24.4, Sep:20.0, Oct:13.3, Nov:6.1, Dec:-1.1
    """
    monthly_avg = {
        1: -3.5, 2: -1.7, 3: 4.4, 4: 10.6, 5: 16.7, 6: 22.2,
        7: 25.0, 8: 24.4, 9: 20.0, 10: 13.3, 11: 6.1, 12: -1.1
    }
    monthly_range = {
        1: 6, 2: 7, 3: 8, 4: 10, 5: 10, 6: 9,
        7: 8, 8: 8, 9: 9, 10: 10, 11: 8, 12: 6
    }

    avg = monthly_avg.get(month, 10.0)
    amp = monthly_range.get(month, 8) / 2
    rng = np.random.RandomState(seed + month)

    def temperature_at_hour(hour: float) -> float:
        # Min temp at 5 AM, max at 3 PM
        base = avg + amp * np.sin(2 * np.pi * (hour - 5) / 24 - np.pi / 2)
        noise = rng.normal(0, 1.5)
        return base + noise

    return temperature_at_hour


# =============================================================================
# FLEET SIMULATOR (Discrete Event Simulation)
# =============================================================================

class FleetSimulator:
    """
    Discrete Event Simulation of an autonomous EV shuttle fleet.

    Architecture:
        - Priority queue of events sorted by time
        - Vehicle state updated after each event
        - Charging station state tracks charger availability and queues
        - Logger records all events for output datasets
    """

    def __init__(
        self,
        pois: pd.DataFrame,
        stations: pd.DataFrame,
        distance_matrix: pd.DataFrame,
        energy_matrix: pd.DataFrame,
        n_vehicles: int = 5,
        soc_charge_threshold: float = 0.30,
        operating_hours: Tuple[int, int] = (7, 23),
        month: int = 3,
        seed: int = 42
    ):
        self.pois = pois
        self.distance_matrix = distance_matrix
        self.energy_matrix = energy_matrix
        self.operating_hours = operating_hours
        self.energy_model = EnergyModel()
        self.vehicle_params = VehicleParams()
        self.soc_threshold = soc_charge_threshold
        self.rng = np.random.RandomState(seed)

        # Temperature profile
        self.temp_func = get_temperature_profile(month, seed)

        # Event queue
        self.event_queue: List[Event] = []
        self.current_time: float = operating_hours[0] * 3600

        # Initialize vehicles at high-demand POIs
        self.vehicles = self._init_vehicles(n_vehicles)

        # Initialize charging stations
        self.stations = self._init_stations(stations)

        # Trip tracking
        self.all_requests: List[TripRequest] = []
        self.pending_queue: List[TripRequest] = []

        # Logging
        self.trip_log: List[dict] = []
        self.charging_log: List[dict] = []
        self.vehicle_state_log: List[dict] = []

    # -----------------------------------------------------------------
    # INITIALIZATION
    # -----------------------------------------------------------------

    def _init_vehicles(self, n: int) -> List[Vehicle]:
        """Initialize vehicles at spread-out high-demand POIs."""
        sorted_pois = self.pois.sort_values('priority').head(n)
        vehicles = []

        for i, (_, poi) in enumerate(sorted_pois.iterrows()):
            v = Vehicle(
                vehicle_id=i,
                lat=poi['latitude'],
                lon=poi['longitude'],
                location_name=poi['poi_name'],
                soc=self.vehicle_params.soc_max
            )
            vehicles.append(v)

        # If fewer POIs than vehicles, distribute remaining
        while len(vehicles) < n:
            idx = len(vehicles) % len(self.pois)
            poi = self.pois.iloc[idx]
            v = Vehicle(
                vehicle_id=len(vehicles),
                lat=poi['latitude'],
                lon=poi['longitude'],
                location_name=poi['poi_name'],
                soc=self.vehicle_params.soc_max
            )
            vehicles.append(v)

        return vehicles

    def _init_stations(self, stations_df: pd.DataFrame) -> Dict[str, ChargingStation]:
        """Initialize charging stations from selected_stations.csv."""
        stations = {}
        # Handle both 'poi_name' and 'station_name' column names
        name_col = 'station_name' if 'station_name' in stations_df.columns else 'poi_name'
        for _, row in stations_df.iterrows():
            stations[row[name_col]] = ChargingStation(
                name=row[name_col],
                lat=row['latitude'],
                lon=row['longitude'],
                n_chargers=int(row.get('charger_count', 2))
            )
        return stations

    # -----------------------------------------------------------------
    # UTILITY METHODS
    # -----------------------------------------------------------------

    def _schedule_event(self, event: Event):
        """Add event to the priority queue."""
        heapq.heappush(self.event_queue, event)

    def _time_to_str(self, seconds: float) -> str:
        """Convert seconds from midnight to HH:MM:SS string."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _distance_between(self, lat1, lon1, lat2, lon2) -> float:
        """Road distance in km."""
        return road_distance(haversine_distance(lat1, lon1, lat2, lon2))

    def _travel_time_sec(self, distance_km: float, speed_kmh: float) -> float:
        """Travel time in seconds."""
        if speed_kmh <= 0 or distance_km <= 0:
            return 0.0
        return (distance_km / speed_kmh) * 3600

    def _log_vehicle_state(self, vehicle: Vehicle, event_desc: str):
        """Log current vehicle state."""
        self.vehicle_state_log.append({
            'time_sec': self.current_time,
            'time_str': self._time_to_str(self.current_time),
            'vehicle_id': vehicle.vehicle_id,
            'status': vehicle.status.value,
            'lat': vehicle.lat,
            'lon': vehicle.lon,
            'location': vehicle.location_name,
            'soc': round(vehicle.soc, 4),
            'km_today': round(vehicle.km_today, 3),
            'trips_served': vehicle.trips_served,
            'event': event_desc,
        })

    # -----------------------------------------------------------------
    # DISPATCH POLICY: Nearest-Available with SoC Check
    # -----------------------------------------------------------------

    def _find_nearest_vehicle(self, request: TripRequest) -> Optional[Vehicle]:
        """Find nearest idle vehicle with sufficient SoC."""
        best_vehicle = None
        best_dist = float('inf')

        for v in self.vehicles:
            if v.status != VehicleStatus.IDLE:
                continue
            if v.soc <= self.soc_threshold:
                continue

            dist = self._distance_between(
                v.lat, v.lon, request.origin_lat, request.origin_lon
            )

            # Check if vehicle has enough energy for deadhead + trip
            trip_dist = self._distance_between(
                request.origin_lat, request.origin_lon,
                request.dest_lat, request.dest_lon
            )
            total_energy_est = self.energy_model.compute_trip_energy(
                distance_km=dist + trip_dist,
                n_passengers=request.n_passengers
            )
            energy_available = (
                (v.soc - self.vehicle_params.soc_min)
                * self.vehicle_params.battery_capacity
            )

            if total_energy_est <= energy_available and dist < best_dist:
                best_dist = dist
                best_vehicle = v

        return best_vehicle

    def _find_nearest_station(self, vehicle: Vehicle) -> Optional[ChargingStation]:
        """Find nearest charging station."""
        best_station = None
        best_dist = float('inf')

        for name, station in self.stations.items():
            dist = self._distance_between(
                vehicle.lat, vehicle.lon, station.lat, station.lon
            )
            if dist < best_dist:
                best_dist = dist
                best_station = station

        return best_station

    def _check_charging_needed(self, vehicle: Vehicle):
        """Check if vehicle needs charging after completing a task."""
        if vehicle.soc <= self.soc_threshold:
            station = self._find_nearest_station(vehicle)
            if station is None:
                return

            speed = 20.0 + self.rng.normal(0, 2)
            speed = max(10, min(30, speed))
            dist = self._distance_between(
                vehicle.lat, vehicle.lon, station.lat, station.lon
            )
            travel_time = self._travel_time_sec(dist, speed)

            energy = self.energy_model.compute_trip_energy(
                distance_km=dist,
                avg_speed_kmh=speed,
                n_passengers=0
            )

            vehicle.status = VehicleStatus.EN_ROUTE_STATION
            self._log_vehicle_state(vehicle, f"heading_to_station_{station.name}")

            self._schedule_event(Event(
                time=self.current_time + travel_time,
                event_type=EventType.VEHICLE_AT_STATION,
                data={
                    'vehicle_id': vehicle.vehicle_id,
                    'station_name': station.name,
                    'distance_km': dist,
                    'energy_kWh': energy,
                    'speed_kmh': speed
                }
            ))
        else:
            # Try to serve pending requests
            self._try_serve_pending(vehicle)

    def _try_serve_pending(self, vehicle: Vehicle):
        """Try to assign a pending request to this vehicle."""
        if not self.pending_queue:
            return

        for i, request in enumerate(self.pending_queue):
            if vehicle.soc <= self.soc_threshold:
                break

            dist_to_pickup = self._distance_between(
                vehicle.lat, vehicle.lon,
                request.origin_lat, request.origin_lon
            )
            trip_dist = self._distance_between(
                request.origin_lat, request.origin_lon,
                request.dest_lat, request.dest_lon
            )
            total_energy = self.energy_model.compute_trip_energy(
                distance_km=dist_to_pickup + trip_dist,
                n_passengers=request.n_passengers
            )
            energy_available = (
                (vehicle.soc - self.vehicle_params.soc_min)
                * self.vehicle_params.battery_capacity
            )

            if total_energy <= energy_available:
                request = self.pending_queue.pop(i)
                self._schedule_event(Event(
                    time=self.current_time,
                    event_type=EventType.TRIP_REQUEST,
                    data={'request': request}
                ))
                return

    # -----------------------------------------------------------------
    # CC-CV CHARGING MODEL
    # -----------------------------------------------------------------

    def _compute_charge_time_sec(
        self, soc_start: float, soc_end: float, rate_kW: float
    ) -> float:
        """
        Compute charging time using CC-CV approximation.

        CC phase: constant power up to 80% SoC
        CV phase: power tapers linearly from 100% to 30% between 80-95% SoC
        """
        cap = self.vehicle_params.battery_capacity
        time_sec = 0.0

        soc = soc_start
        step = 0.01

        while soc < soc_end - 1e-6:
            next_soc = min(soc + step, soc_end)
            energy_needed = (next_soc - soc) * cap

            if soc < 0.80:
                effective_rate = rate_kW
            else:
                fraction = (soc - 0.80) / (0.95 - 0.80)
                effective_rate = rate_kW * (1.0 - 0.7 * fraction)

            time_sec += (energy_needed / effective_rate) * 3600
            soc = next_soc

        return time_sec

    # -----------------------------------------------------------------
    # EVENT HANDLERS
    # -----------------------------------------------------------------

    def _handle_trip_request(self, event: Event):
        """Handle a new trip request."""
        request = event.data['request']

        vehicle = self._find_nearest_vehicle(request)

        if vehicle is None:
            request.status = "pending"
            self.pending_queue.append(request)
            return

        request.assigned_vehicle = vehicle.vehicle_id
        request.status = "assigned"
        vehicle.current_trip_id = request.trip_id

        deadhead_dist = self._distance_between(
            vehicle.lat, vehicle.lon,
            request.origin_lat, request.origin_lon
        )

        speed_pickup = 20.0 + self.rng.normal(0, 3)
        speed_pickup = max(10, min(30, speed_pickup))
        travel_time = self._travel_time_sec(deadhead_dist, speed_pickup)

        hour = self.current_time / 3600
        temp = self.temp_func(hour)

        energy_deadhead = self.energy_model.compute_trip_energy(
            distance_km=deadhead_dist,
            avg_speed_kmh=speed_pickup,
            n_passengers=0,
            n_stops=0,
            temperature_C=temp
        )

        vehicle.status = VehicleStatus.EN_ROUTE_PICKUP
        self._log_vehicle_state(vehicle, f"dispatched_to_{request.origin_name}")

        self._schedule_event(Event(
            time=self.current_time + travel_time,
            event_type=EventType.VEHICLE_AT_PICKUP,
            data={
                'vehicle_id': vehicle.vehicle_id,
                'request': request,
                'deadhead_km': deadhead_dist,
                'deadhead_energy': energy_deadhead,
                'deadhead_speed': speed_pickup,
                'temperature_C': temp
            }
        ))

    def _handle_vehicle_at_pickup(self, event: Event):
        """Vehicle has arrived at pickup location."""
        vehicle = self.vehicles[event.data['vehicle_id']]
        request = event.data['request']
        deadhead_energy = event.data['deadhead_energy']
        deadhead_km = event.data['deadhead_km']
        temp = event.data['temperature_C']

        # Consume deadhead energy
        vehicle.soc -= deadhead_energy / self.vehicle_params.battery_capacity
        vehicle.km_today += deadhead_km
        vehicle.energy_consumed_kWh += deadhead_energy

        # Update location
        vehicle.lat = request.origin_lat
        vehicle.lon = request.origin_lon
        vehicle.location_name = request.origin_name

        # Pick up passengers
        vehicle.passengers = request.n_passengers
        request.pickup_time = self.current_time
        request.wait_time_sec = request.pickup_time - request.request_time

        # Compute trip to destination
        trip_dist = self._distance_between(
            request.origin_lat, request.origin_lon,
            request.dest_lat, request.dest_lon
        )

        speed_trip = 20.0 + self.rng.normal(0, 3)
        speed_trip = max(10, min(30, speed_trip))
        grade = self.rng.normal(0, 0.005)
        grade = max(-0.03, min(0.03, grade))
        n_stops = self.rng.poisson(1)

        travel_time = self._travel_time_sec(trip_dist, speed_trip)

        energy_trip = self.energy_model.compute_trip_energy(
            distance_km=trip_dist,
            avg_speed_kmh=speed_trip,
            n_passengers=request.n_passengers,
            n_stops=n_stops,
            grade_avg=grade,
            temperature_C=temp
        )

        vehicle.status = VehicleStatus.EN_ROUTE_DROPOFF
        self._log_vehicle_state(vehicle, f"pickup_{request.origin_name}_to_{request.dest_name}")

        self._schedule_event(Event(
            time=self.current_time + travel_time,
            event_type=EventType.VEHICLE_AT_DROPOFF,
            data={
                'vehicle_id': vehicle.vehicle_id,
                'request': request,
                'trip_distance_km': trip_dist,
                'trip_energy_kWh': energy_trip,
                'trip_speed_kmh': speed_trip,
                'trip_grade': grade,
                'trip_n_stops': n_stops,
                'temperature_C': temp,
                'deadhead_km': deadhead_km,
                'deadhead_energy': deadhead_energy
            }
        ))

    def _handle_vehicle_at_dropoff(self, event: Event):
        """Vehicle has arrived at dropoff — log the completed trip."""
        vehicle = self.vehicles[event.data['vehicle_id']]
        request = event.data['request']
        trip_energy = event.data['trip_energy_kWh']
        trip_dist = event.data['trip_distance_km']

        # Consume trip energy
        vehicle.soc -= trip_energy / self.vehicle_params.battery_capacity
        vehicle.soc = max(vehicle.soc, 0.0)
        vehicle.km_today += trip_dist
        vehicle.energy_consumed_kWh += trip_energy

        # Update vehicle state
        vehicle.lat = request.dest_lat
        vehicle.lon = request.dest_lon
        vehicle.location_name = request.dest_name
        vehicle.passengers = 0
        vehicle.trips_served += 1
        vehicle.current_trip_id = None

        request.dropoff_time = self.current_time
        request.status = "completed"

        vehicle.status = VehicleStatus.IDLE
        self._log_vehicle_state(vehicle, f"dropoff_{request.dest_name}")

        # Log completed trip (this becomes ML training data)
        hour = request.request_time / 3600
        self.trip_log.append({
            'trip_id': request.trip_id,
            'vehicle_id': vehicle.vehicle_id,
            'origin': request.origin_name,
            'origin_lat': request.origin_lat,
            'origin_lon': request.origin_lon,
            'destination': request.dest_name,
            'dest_lat': request.dest_lat,
            'dest_lon': request.dest_lon,
            'distance_km': round(trip_dist, 4),
            'n_passengers': request.n_passengers,
            'avg_speed_kmh': round(event.data['trip_speed_kmh'], 2),
            'grade_avg': round(event.data['trip_grade'], 5),
            'n_stops': event.data['trip_n_stops'],
            'temperature_C': round(event.data['temperature_C'], 2),
            'hour': int(hour),
            'timestamp_sec': request.request_time,
            'wait_time_sec': round(request.wait_time_sec, 1),
            'energy_kWh': round(trip_energy, 6),
            'deadhead_km': round(event.data['deadhead_km'], 4),
            'deadhead_energy_kWh': round(event.data['deadhead_energy'], 6),
            'soc_after': round(vehicle.soc, 4),
        })

        # Check if charging needed, or serve pending
        self._check_charging_needed(vehicle)

    def _handle_vehicle_at_station(self, event: Event):
        """Vehicle has arrived at charging station."""
        vehicle = self.vehicles[event.data['vehicle_id']]
        station = self.stations[event.data['station_name']]
        energy_to_station = event.data['energy_kWh']
        dist_to_station = event.data['distance_km']

        # Consume travel energy
        vehicle.soc -= energy_to_station / self.vehicle_params.battery_capacity
        vehicle.soc = max(vehicle.soc, 0.0)
        vehicle.km_today += dist_to_station
        vehicle.energy_consumed_kWh += energy_to_station

        # Update location
        vehicle.lat = station.lat
        vehicle.lon = station.lon
        vehicle.location_name = station.name

        if station.has_available_charger:
            self._start_charging(vehicle, station)
        else:
            vehicle.status = VehicleStatus.WAITING_CHARGER
            station.queue.append(vehicle.vehicle_id)
            self._log_vehicle_state(vehicle, f"queuing_at_{station.name}")

    def _start_charging(self, vehicle: Vehicle, station: ChargingStation):
        """Begin a charging session."""
        station.vehicles_charging.append(vehicle.vehicle_id)
        vehicle.status = VehicleStatus.CHARGING

        soc_start = vehicle.soc
        soc_end = self.vehicle_params.soc_max
        charge_time = self._compute_charge_time_sec(
            soc_start, soc_end, station.charging_rate_kW
        )

        self._log_vehicle_state(vehicle, f"charging_start_at_{station.name}")

        self._schedule_event(Event(
            time=self.current_time + charge_time,
            event_type=EventType.CHARGING_COMPLETE,
            data={
                'vehicle_id': vehicle.vehicle_id,
                'station_name': station.name,
                'soc_start': soc_start,
                'soc_end': soc_end,
                'charge_time_sec': charge_time,
                'start_time': self.current_time
            }
        ))

    def _handle_charging_complete(self, event: Event):
        """Vehicle finished charging."""
        vehicle = self.vehicles[event.data['vehicle_id']]
        station = self.stations[event.data['station_name']]

        vehicle.soc = event.data['soc_end']
        vehicle.status = VehicleStatus.IDLE

        # Remove from station
        if vehicle.vehicle_id in station.vehicles_charging:
            station.vehicles_charging.remove(vehicle.vehicle_id)

        # Log charging event
        self.charging_log.append({
            'vehicle_id': vehicle.vehicle_id,
            'station': station.name,
            'start_time_sec': event.data['start_time'],
            'end_time_sec': self.current_time,
            'duration_sec': event.data['charge_time_sec'],
            'soc_start': round(event.data['soc_start'], 4),
            'soc_end': round(event.data['soc_end'], 4),
            'energy_charged_kWh': round(
                (event.data['soc_end'] - event.data['soc_start'])
                * self.vehicle_params.battery_capacity, 4
            ),
        })

        self._log_vehicle_state(vehicle, f"charging_complete_at_{station.name}")

        # Start charging next vehicle in queue
        if station.queue:
            next_vid = station.queue.pop(0)
            next_vehicle = self.vehicles[next_vid]
            self._start_charging(next_vehicle, station)

        # Try to serve pending requests
        self._try_serve_pending(vehicle)

    # -----------------------------------------------------------------
    # MAIN SIMULATION LOOP
    # -----------------------------------------------------------------

    def run(self, requests: List[TripRequest] = None, verbose: bool = True) -> Dict:
        """
        Run the full-day simulation.

        Parameters
        ----------
        requests : list of TripRequest, optional
            Pre-generated trip requests. If None, generates automatically.
        verbose : bool
            Print progress updates.

        Returns
        -------
        dict with keys: trip_data, charging_events, vehicle_logs, summary
        """
        if verbose:
            print("\n" + "=" * 60)
            print("FLEET SIMULATION — Discrete Event Simulation")
            print("=" * 60)

        # Generate requests if not provided
        if requests is None:
            if verbose:
                print("\nGenerating trip requests (Poisson process)...")
            demand_gen = DemandGenerator(self.pois)
            requests = demand_gen.generate_requests()

        self.all_requests = requests
        if verbose:
            print(f"  Total requests: {len(requests)}")

        # Schedule all trip request events
        for req in requests:
            self._schedule_event(Event(
                time=req.request_time,
                event_type=EventType.TRIP_REQUEST,
                data={'request': req}
            ))

        # Log initial vehicle states
        for v in self.vehicles:
            self._log_vehicle_state(v, "initial")

        # --- Main event loop ---
        if verbose:
            print("\nRunning simulation...")
            total_events = len(self.event_queue)
            milestone = max(1, total_events // 10)

        events_processed = 0
        end_time = (self.operating_hours[1] + 1) * 3600

        while self.event_queue:
            event = heapq.heappop(self.event_queue)

            if event.time > end_time:
                break

            self.current_time = event.time

            if event.event_type == EventType.TRIP_REQUEST:
                self._handle_trip_request(event)
            elif event.event_type == EventType.VEHICLE_AT_PICKUP:
                self._handle_vehicle_at_pickup(event)
            elif event.event_type == EventType.VEHICLE_AT_DROPOFF:
                self._handle_vehicle_at_dropoff(event)
            elif event.event_type == EventType.VEHICLE_AT_STATION:
                self._handle_vehicle_at_station(event)
            elif event.event_type == EventType.CHARGING_COMPLETE:
                self._handle_charging_complete(event)

            events_processed += 1
            if verbose and events_processed % milestone == 0:
                pct = (events_processed / max(total_events, 1)) * 100
                print(f"  {pct:.0f}% — {self._time_to_str(self.current_time)}")

        # Mark remaining pending requests as missed
        for req in self.pending_queue:
            req.status = "missed"

        # --- Compile results ---
        trip_data = pd.DataFrame(self.trip_log)
        charging_events = pd.DataFrame(self.charging_log)
        vehicle_logs = pd.DataFrame(self.vehicle_state_log)

        # Summary
        n_completed = len(trip_data)
        n_missed = sum(1 for r in self.all_requests if r.status == "missed")
        n_pending = sum(1 for r in self.all_requests if r.status == "pending")

        summary = {
            'total_requests': len(self.all_requests),
            'trips_completed': n_completed,
            'trips_missed': n_missed + n_pending,
            'service_rate': round(n_completed / max(len(self.all_requests), 1), 4),
            'total_distance_km': round(sum(v.km_today for v in self.vehicles), 2),
            'total_energy_kWh': round(sum(v.energy_consumed_kWh for v in self.vehicles), 2),
            'charging_sessions': len(self.charging_log),
            'avg_wait_time_sec': round(trip_data['wait_time_sec'].mean(), 1) if len(trip_data) > 0 else 0,
            'n_vehicles': len(self.vehicles),
            'operating_hours': f"{self.operating_hours[0]}:00 - {self.operating_hours[1]}:00",
            'vehicle_stats': [
                {
                    'vehicle_id': v.vehicle_id,
                    'trips_served': v.trips_served,
                    'km_driven': round(v.km_today, 2),
                    'energy_consumed_kWh': round(v.energy_consumed_kWh, 2),
                    'final_soc': round(v.soc, 4),
                }
                for v in self.vehicles
            ],
        }

        if verbose:
            print(f"\n{'=' * 60}")
            print("SIMULATION COMPLETE")
            print(f"{'=' * 60}")
            print(f"  Trips completed: {n_completed}/{len(self.all_requests)} "
                  f"({summary['service_rate']:.1%})")
            print(f"  Trips missed: {n_missed + n_pending}")
            print(f"  Total distance: {summary['total_distance_km']:.1f} km")
            print(f"  Total energy: {summary['total_energy_kWh']:.1f} kWh")
            print(f"  Charging sessions: {summary['charging_sessions']}")
            print(f"  Avg wait time: {summary['avg_wait_time_sec']:.0f} sec")

        return {
            'trip_data': trip_data,
            'charging_events': charging_events,
            'vehicle_logs': vehicle_logs,
            'summary': summary,
        }


# =============================================================================
# MAIN
# =============================================================================

def main(data_dir: str = None, output_dir: str = None) -> Dict:
    """
    Run fleet simulation and save outputs.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing input files (selected_pois.csv, selected_stations.csv,
        distance_matrix.csv, energy_matrix.csv). Defaults to 01_Data_gen/Outputs.
    output_dir : str, optional
        Directory for output files. Defaults to 01_Data_gen/Outputs.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "01_Data_gen" / "Outputs"
    base_dir = Path(data_dir)
    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        # Default: save under 01_Data_gen/Outputs (relative to this script)
        out_dir = Path(__file__).resolve().parent / "01_Data_gen" / "Outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs (selected_pois.csv may live in parent dir, e.g. 01_Data_gen/ vs 01_Data_gen/Outputs)
    pois_path = base_dir / 'selected_pois.csv'
    if not pois_path.exists():
        pois_path = base_dir.parent / 'selected_pois.csv'
    if not pois_path.exists():
        raise FileNotFoundError(
            f"selected_pois.csv not found in {base_dir} or {base_dir.parent}. "
            "Run 01_Data_gen/poi_selection.py first, then station_data.py and station_selection.py."
        )
    pois = pd.read_csv(pois_path)
    stations = pd.read_csv(base_dir / 'selected_stations.csv')
    distance_matrix = pd.read_csv(base_dir / 'distance_matrix.csv', index_col=0)
    energy_matrix = pd.read_csv(base_dir / 'energy_matrix.csv', index_col=0)

    # Load fleet size from summary if available
    n_vehicles = 10
    summary_path = base_dir / 'summary_stats.json'
    if summary_path.exists():
        with open(summary_path) as f:
            stats = json.load(f)
            n_vehicles = stats.get('estimated_fleet_size', 10)

    print(f"Fleet size: {n_vehicles} vehicles")

    # Create simulator
    sim = FleetSimulator(
        pois=pois,
        stations=stations,
        distance_matrix=distance_matrix,
        energy_matrix=energy_matrix,
        n_vehicles=n_vehicles,
        soc_charge_threshold=0.30,
        operating_hours=(7, 23),
        month=3,
        seed=42
    )

    # Run simulation
    results = sim.run(verbose=True)

    # Save outputs
    print(f"\n{'=' * 60}")
    print("SAVING OUTPUTS")
    print(f"{'=' * 60}")

    trip_path = out_dir / "trip_data.csv"
    results['trip_data'].to_csv(trip_path, index=False)
    print(f"  trip_data.csv — {len(results['trip_data'])} trips")

    charge_path = out_dir / "charging_events.csv"
    results['charging_events'].to_csv(charge_path, index=False)
    print(f"  charging_events.csv — {len(results['charging_events'])} sessions")

    log_path = out_dir / "vehicle_logs.csv"
    results['vehicle_logs'].to_csv(log_path, index=False)
    print(f"  vehicle_logs.csv — {len(results['vehicle_logs'])} state records")

    summary_path = out_dir / "fleet_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results['summary'], f, indent=2)
    print(f"  fleet_summary.json — daily summary")

    print(f"\n✓ Fleet simulation complete — datasets ready for ML training")

    return results


if __name__ == "__main__":
    results = main()