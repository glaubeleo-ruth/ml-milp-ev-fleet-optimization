"""
Real-time single-scenario simulator.

Runs one rolling-horizon scenario (MILP or greedy) and optionally updates
a live map each epoch so you can watch vehicles move. Useful for demos
and debugging.

Usage:
  python realtime_sim.py                    # live map, greedy, 0.2 s/epoch
  python realtime_sim.py --strategy milp    # MILP dispatch (slower)
  python realtime_sim.py --delay 0.5        # slower playback
  python realtime_sim.py --no-plot          # terminal progress only, no window
  python realtime_sim.py --max-epochs 50    # run first 50 epochs only (quick test)
  python realtime_sim.py --vehicle-icon car.png --station-icon charger.png  # custom PNG icons
  # Default: 05_Scenario/icons/bus.png (shuttle), icons/person.png (station) if present.

Requires: matplotlib. Optional basemap: pip install contextily
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Import rolling horizon from 03_MILP
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent / "03_MILP"))
from rolling_horizon import RollingHorizonController

# Same station/vehicle config as rolling_horizon main
STATION_INFO = [
    {'name': 'MTCC', 'lat': 41.838385, 'lon': -87.627555, 'chargers': 2},
    {'name': 'Paul Galvin Library', 'lat': 41.833675, 'lon': -87.628336, 'chargers': 2},
    {'name': 'McCormick Student Village', 'lat': 41.835527, 'lon': -87.624207, 'chargers': 2},
    {'name': 'Crown Hall', 'lat': 41.833199, 'lon': -87.627273, 'chargers': 2},
    {'name': 'Kaplan Institute', 'lat': 41.836861, 'lon': -87.628300, 'chargers': 2},
    {'name': 'Arthur S. Keating Sports Center', 'lat': 41.838985, 'lon': -87.625566, 'chargers': 2},
]
VEHICLE_STARTS = [
    {'id': i, 'lat': lat, 'lon': lon}
    for i, (lat, lon) in [
        (0, (41.837866, -87.624703)), (1, (41.831394, -87.627231)),
        (2, (41.833199, -87.627273)), (3, (41.835681, -87.628387)),
        (4, (41.836861, -87.628300)), (5, (41.835527, -87.624207)),
        (6, (41.833675, -87.628336)), (7, (41.834344, -87.623795)),
        (8, (41.834344, -87.623795)), (9, (41.837866, -87.624703)),
    ]
]


def _resolve_trip_data(base_dir: Path) -> Path:
    candidates = [
        base_dir / 'trip_data.csv',
        base_dir / '01_Data_gen' / 'Outputs' / 'trip_data.csv',
        _here.parent / '01_Data_gen' / 'Outputs' / 'trip_data.csv',
        _here / 'trip_data.csv',
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("trip_data.csv not found. Tried: " + ", ".join(str(p) for p in candidates))


def _fmt_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    return f"{h:02d}:{m:02d}"


def main():
    parser = argparse.ArgumentParser(description='Real-time single-scenario simulator')
    parser.add_argument('--strategy', choices=['milp', 'greedy'], default='greedy',
                        help='Dispatch strategy (greedy is faster for live view)')
    parser.add_argument('--delay', type=float, default=0.6,
                        help='Seconds to pause per epoch (default 0.6; 0 = no delay)')
    parser.add_argument('--no-plot', action='store_true', help='No live map, terminal only')
    parser.add_argument('--max-epochs', type=int, default=None, help='Stop after N epochs (for testing)')
    parser.add_argument('--vehicles', type=int, default=10)
    parser.add_argument('--epoch-min', type=float, default=5.0, help='Epoch length (minutes)')
    parser.add_argument('--data-dir', type=str, default='.')
    _icons_dir = _here / 'icons'
    _default_bus = _icons_dir / 'bus.png'      # shuttle/vehicle icon
    _default_person = _icons_dir / 'person.png'  # station icon
    parser.add_argument('--vehicle-icon', type=str, default=None,
                        help=f'Path to PNG for shuttle/vehicle (default: {_default_bus.name} if present)')
    parser.add_argument('--station-icon', type=str, default=None,
                        help=f'Path to PNG for station (default: {_default_person.name} if present)')
    parser.add_argument('--icon-size', type=float, default=0.018,
                        help='Icon size as fraction of map (default 0.018); tune if icons too big/small')
    args = parser.parse_args()
    # Use project icons when no path given and defaults exist
    vehicle_icon_path = args.vehicle_icon or (str(_default_bus) if _default_bus.exists() else None)
    station_icon_path = args.station_icon or (str(_default_person) if _default_person.exists() else None)

    base_dir = Path(args.data_dir)
    trip_path = _resolve_trip_data(base_dir)
    trip_data = pd.read_csv(trip_path)
    print(f"Loaded {len(trip_data)} trips from {trip_path}")

    ml_path = str(base_dir / 'trained_model.pkl') if (base_dir / 'trained_model.pkl').exists() else None
    rh = RollingHorizonController(
        trip_data=trip_data,
        station_info=STATION_INFO,
        vehicle_starts=VEHICLE_STARTS,
        ml_model_path=ml_path,
        epoch_minutes=args.epoch_min,
        n_vehicles=args.vehicles,
    )

    fig, ax = None, None
    scatter_veh = None
    vehicle_icon_boxes = None  # list of AnnotationBbox when using PNG vehicle icon
    veh_im = None  # loaded bus image for redraw-each-epoch path
    icon_zoom = None
    n_epochs_total = int(
        (trip_data['timestamp_sec'].max() + 300 - (trip_data['timestamp_sec'].min() - 60))
        / (args.epoch_min * 60)
    )
    max_epochs = args.max_epochs if args.max_epochs is not None else n_epochs_total

    def on_epoch(epoch_idx: int, current_time: float, vehicles: list, n_served: int, n_pending: int):
        if epoch_idx >= max_epochs:
            return
        if not args.no_plot and fig is not None and ax is not None:
            lats = [v['lat'] for v in vehicles]
            lons = [v['lon'] for v in vehicles]
            if vehicle_icon_boxes is not None and veh_im is not None and icon_zoom is not None:
                # Re-draw vehicle icons at new positions (AnnotationBbox.xy update is unreliable)
                for box in vehicle_icon_boxes:
                    box.remove()
                vehicle_icon_boxes.clear()
                for i in range(len(lons)):
                    imbox = AnnotationBbox(
                        OffsetImage(veh_im, zoom=icon_zoom),
                        (lons[i], lats[i]), frameon=False
                    )
                    ax.add_artist(imbox)
                    vehicle_icon_boxes.append(imbox)
            else:
                scatter_veh.set_offsets(list(zip(lons, lats)))
            ax.set_title(f"Epoch {epoch_idx}/{max_epochs} — {_fmt_time(current_time)} — "
                         f"served: {n_served}, pending: {n_pending}")
            fig.canvas.draw()
            fig.canvas.flush_events()
        if args.delay > 0:
            time.sleep(args.delay)

    if not args.no_plot:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox

        fig, ax = plt.subplots(figsize=(10, 8))
        margin = 0.002
        lon_min = trip_data['origin_lon'].min() - margin
        lon_max = trip_data['origin_lon'].max() + margin
        lat_min = trip_data['origin_lat'].min() - margin
        lat_max = trip_data['origin_lat'].max() + margin
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_aspect('equal')
        try:
            import contextily as ctx
            ctx.add_basemap(ax, crs='EPSG:4326', zoom=17)
        except Exception:
            ax.grid(True, alpha=0.3)
        stn_lons = [s['lon'] for s in STATION_INFO]
        stn_lats = [s['lat'] for s in STATION_INFO]

        # Icon zoom: scale so icons fit map (smaller axis span → smaller icons)
        data_span = min(lon_max - lon_min, lat_max - lat_min)
        icon_zoom = args.icon_size * 1.2 * (0.01 / max(data_span, 0.005))
        stn_icon_zoom = icon_zoom * 2.2  # person/station icon larger than bus

        # Stations: PNG icon (person) or default scatter
        if station_icon_path and Path(station_icon_path).exists():
            stn_im = mpimg.imread(station_icon_path)
            for sx, sy in zip(stn_lons, stn_lats):
                imbox = AnnotationBbox(
                    OffsetImage(stn_im, zoom=stn_icon_zoom),
                    (sx, sy), frameon=False
                )
                ax.add_artist(imbox)
        else:
            ax.scatter(stn_lons, stn_lats, c='green', s=100, marker='^', label='Stations', zorder=5)

        # Vehicles: PNG icon (bus/shuttle) or default scatter
        if vehicle_icon_path and Path(vehicle_icon_path).exists():
            veh_im = mpimg.imread(vehicle_icon_path)
            vehicle_icon_boxes = []
            for v in VEHICLE_STARTS[:args.vehicles]:
                imbox = AnnotationBbox(
                    OffsetImage(veh_im, zoom=icon_zoom),
                    (v['lon'], v['lat']), frameon=False
                )
                ax.add_artist(imbox)
                vehicle_icon_boxes.append(imbox)
        else:
            veh_im = None
            icon_zoom = None
            scatter_veh = ax.scatter(
                [v['lon'] for v in VEHICLE_STARTS[:args.vehicles]],
                [v['lat'] for v in VEHICLE_STARTS[:args.vehicles]],
                c='red', s=80, edgecolors='white', linewidths=1, zorder=6, label='Vehicles',
            )

        ax.legend()
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.ion()
        plt.show(block=False)

    print(f"Running real-time sim: strategy={args.strategy}, max_epochs={max_epochs}, delay={args.delay}s")
    try:
        results = rh.run(
            strategy=args.strategy,
            verbose=True,
            on_epoch_end=on_epoch if (not args.no_plot or args.delay > 0) else None,
            max_epochs=args.max_epochs,
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
        results = None

    if not args.no_plot and fig is not None:
        plt.ioff()
        plt.show(block=True)

    if results:
        print(f"Service rate: {results['service_rate_pct']:.1f}%, "
              f"served {results['trips_served']}/{results['total_trips']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
