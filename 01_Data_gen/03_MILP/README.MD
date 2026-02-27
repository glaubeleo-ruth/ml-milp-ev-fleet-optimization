### MILP Formulation — EV Shuttle Fleet Dispatch & Charging
========================================================

Mixed-Integer Linear Program for jointly optimizing vehicle-trip
assignment and charging decisions in an autonomous electric shuttle fleet.

Solver: PuLP (Gurobi backend, CBC fallback)

Pipeline Position: Stage 5 (MILP Optimization)
    trained_model.pkl + trip_data.csv + selected_stations.csv
        → [THIS SCRIPT] → per-epoch dispatch + charging decisions

Formulation
-----------

SETS
    V   = available (idle) vehicles          indexed by v
    T   = pending trip requests              indexed by t
    S   = charging stations                  indexed by s

PARAMETERS
    e_trip[v,t]     predicted energy (kWh) for vehicle v to serve trip t
                    (includes deadhead energy to reach pickup)
    w[v,t]          estimated wait time (sec) if vehicle v serves trip t
                    (deadhead travel time from current location to pickup)
    soc[v]          current state-of-charge of vehicle v  (0–1)
    cap             battery capacity (kWh)
    soc_min         minimum operating SoC (below this → must charge)
    e_station[v,s]  energy (kWh) to drive from vehicle v to station s
    avail[s]        number of available chargers at station s (0, 1, or 2)
    feasible[v,t]   1 if vehicle v has enough energy to serve trip t
                    pre-computed: soc[v]*cap - e_trip[v,t] ≥ soc_min*cap
    needs_charge[v] 1 if vehicle v's SoC ≤ charge trigger threshold

DECISION VARIABLES
    x[v,t] ∈ {0,1}   1 if vehicle v is assigned to trip t
    c[v,s] ∈ {0,1}   1 if vehicle v is sent to charge at station s

OBJECTIVE
    min   α · Σ_{v,t} e_trip[v,t] · x[v,t]      (total energy)
        + β · Σ_{v,t} w[v,t] · x[v,t]            (total wait time)
        + γ · Σ_t (1 - Σ_v x[v,t])               (unserved trip penalty)

    where α, β, γ are weighting coefficients (normalized by typical values)

CONSTRAINTS
    (C1) Trip assignment:      Σ_v x[v,t] ≤ 1              ∀ t ∈ T
         Each trip assigned to at most one vehicle.

    (C2) Vehicle capacity:     Σ_t x[v,t] + Σ_s c[v,s] ≤ 1  ∀ v ∈ V
         Each vehicle does at most one action per decision epoch.

    (C3) Energy feasibility:   x[v,t] ≤ feasible[v,t]       ∀ v,t
         Vehicle can only serve trip if it has enough energy.
         Pre-filtered: feasible[v,t] = 1 iff
           soc[v] · cap − e_trip[v,t] ≥ soc_min · cap

    (C4) Station capacity:     Σ_v c[v,s] ≤ avail[s]        ∀ s ∈ S
         Don't exceed available chargers at each station.

    (C5) Charge eligibility:   Σ_s c[v,s] ≤ needs_charge[v]  ∀ v ∈ V
         Only vehicles below charge threshold can be sent to charge.

    (C6) Station reachability: c[v,s] · e_station[v,s] ≤ (soc[v]−soc_min)·cap
         Linearized via pre-filter (same logic as C3).

ML Integration:
    The trained model (XGBoost or Random Forest) enters the MILP only
    through pre-computed parameters e_trip, e_station, feasible, reachable.
    The MILP itself remains a standard binary integer program.

Model Size (typical: 10 vehicles × 20 trips × 6 stations):
    Binary variables: 260
    Constraints: 306
    Solve time: < 1 second

References:
    - Bertsimas et al. (2019) — Online vehicle routing
    - Bongiovanni et al. (2019) — EV dial-a-ride MILP
    - Pelletier et al. (2017) — EV fleet charging optimization
