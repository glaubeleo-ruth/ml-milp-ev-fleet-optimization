### Fleet Simulator — Discrete Event Simulation for EV Shuttle Operations

Simulates a full operating day of an autonomous electric shuttle fleet
on the IIT campus. Vehicles receive trip requests, serve passengers,
consume energy, and charge at the 6 selected stations.

Method: Discrete Event Simulation (DES)
    Events are processed in chronological order via a priority queue.
    The simulation jumps from event to event rather than stepping through
    fixed time increments, making it efficient and precise.
    (Banks et al., 2010 — Discrete-Event System Simulation)

Dispatch Policy: Nearest-Available Vehicle with SoC Check
    (Bischoff & Maciejewski, 2016; Alonso-Mora et al., 2017)

Charging Logic: Threshold-based (charge when SoC < 30%)
    CC-CV charging curve for lithium-ion batteries.
    (Pelletier et al., 2017)

Demand Model: Inhomogeneous Poisson Process
    Arrival rate varies by hour and OD category.
    (Ross, 2014 — Introduction to Probability Models)

Pipeline Position: Stage 3 (Fleet Simulation / Dataset Generation)
    selected_pois.csv + selected_stations.csv + matrices
        → [THIS SCRIPT] → trip_data.csv, charging_events.csv,
                           vehicle_logs.csv, fleet_summary.json

Inputs (from Station_data.py and Station_selection.py):
    - selected_pois.csv
    - selected_stations.csv
    - energy_matrix.csv
    - distance_matrix.csv

Outputs (for ML training and MILP validation):
    - trip_data.csv           (~4,000 individual trips with features + energy labels)
    - charging_events.csv     (charging sessions with timing and SoC)
    - vehicle_logs.csv        (full vehicle state trace)
    - fleet_summary.json      (daily aggregate statistics)