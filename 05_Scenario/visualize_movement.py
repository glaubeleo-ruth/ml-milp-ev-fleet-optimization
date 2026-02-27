"""
Visualize vehicle movement from rolling horizon outputs.

Reads *_vehicle_trace.csv (and optionally stations) and plots:
  - Vehicle trails (lat/lon over time), one color per vehicle
  - Charging stations as markers
  - Optional: OpenStreetMap basemap (roads, buildings)
  - Optional: animation over time

Requires: matplotlib. For basemap (roads/buildings): pip install contextily

Usage:
  python visualize_movement.py [trace_csv] [--animate] [--output plot.png]
  python visualize_movement.py ../03_MILP/milp_vehicle_trace.csv
  python visualize_movement.py ../03_MILP/milp_vehicle_trace.csv --output map.png   # with OSM basemap
  python visualize_movement.py ../03_MILP/milp_vehicle_trace.csv --no-basemap        # plain axes only
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Station positions (same as rolling_horizon / scenario_execution)
STATIONS = [
    {'name': 'MTCC', 'lat': 41.838385, 'lon': -87.627555},
    {'name': 'Paul Galvin Library', 'lat': 41.833675, 'lon': -87.628336},
    {'name': 'McCormick Student Village', 'lat': 41.835527, 'lon': -87.624207},
    {'name': 'Crown Hall', 'lat': 41.833199, 'lon': -87.627273},
    {'name': 'Kaplan Institute', 'lat': 41.836861, 'lon': -87.628300},
    {'name': 'Arthur S. Keating Sports Center', 'lat': 41.838985, 'lon': -87.625566},
]


def load_trace(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ('time_sec', 'vehicle_id', 'lat', 'lon'):
        if col not in df.columns:
            raise ValueError(f"Trace CSV must have columns: time_sec, vehicle_id, lat, lon. Got: {list(df.columns)}")
    return df.sort_values(['vehicle_id', 'time_sec']).reset_index(drop=True)


def _add_basemap(ax, lon_min: float, lon_max: float, lat_min: float, lat_max: float, zoom: int = 17) -> bool:
    """Add OSM basemap (roads, buildings). Returns True if successful."""
    try:
        import contextily as ctx
    except ImportError:
        return False
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect('equal')
    try:
        ctx.add_basemap(ax, crs='EPSG:4326', zoom=zoom)
        return True
    except Exception:
        return False


def plot_static(trace: pd.DataFrame, stations: list, ax, title: str = "Vehicle movement", use_basemap: bool = True):
    """Draw vehicle trails and stations on ax (static map). Optionally add OSM basemap."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required. Install with: pip install matplotlib")

    margin = 0.0015
    lon_min = trace['lon'].min() - margin
    lon_max = trace['lon'].max() + margin
    lat_min = trace['lat'].min() - margin
    lat_max = trace['lat'].max() + margin

    if use_basemap and _add_basemap(ax, lon_min, lon_max, lat_min, lat_max):
        pass  # basemap drawn; trails/stations go on top
    else:
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_aspect('equal')

    n_veh = trace['vehicle_id'].nunique()
    try:
        cmap = plt.colormaps['tab10'].resampled(max(n_veh, 10))
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap('tab10', max(n_veh, 10))

    for i, vid in enumerate(sorted(trace['vehicle_id'].unique())):
        sub = trace[trace['vehicle_id'] == vid]
        sub = sub.sort_values('time_sec')
        ax.plot(
            sub['lon'].values,
            sub['lat'].values,
            color=cmap(i % 10),
            alpha=0.7,
            linewidth=1.5,
            label=f'V{vid}',
        )
        # Start marker
        ax.scatter(
            sub['lon'].iloc[0],
            sub['lat'].iloc[0],
            c=[cmap(i % 10)],
            s=40,
            marker='o',
            edgecolors='k',
            linewidths=0.5,
            zorder=5,
        )
        # End marker
        ax.scatter(
            sub['lon'].iloc[-1],
            sub['lat'].iloc[-1],
            c=[cmap(i % 10)],
            s=80,
            marker='s',
            edgecolors='k',
            linewidths=0.5,
            zorder=5,
        )

    # Stations
    stn_lats = [s['lat'] for s in stations]
    stn_lons = [s['lon'] for s in stations]
    ax.scatter(stn_lons, stn_lats, c='green', s=120, marker='^', edgecolors='darkgreen',
               linewidths=1, label='Stations', zorder=6)
    for s in stations:
        ax.annotate(s['name'][:8], (s['lon'], s['lat']), fontsize=6, ha='center', xytext=(0, 8), textcoords='offset points')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', ncol=2, fontsize=7)
    if not use_basemap:
        ax.grid(True, alpha=0.3)


def plot_animated(trace: pd.DataFrame, stations: list, output_path: Path, title: str = "Vehicle movement", use_basemap: bool = True):
    """Animate vehicle positions over time, save to GIF or show."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import PillowWriter, FuncAnimation
    except ImportError:
        raise ImportError("matplotlib is required. Install with: pip install matplotlib")

    times = sorted(trace['time_sec'].unique())
    if len(times) > 150:
        # Subsample to avoid huge GIFs
        step = max(1, len(times) // 120)
        times = times[::step]

    margin = 0.0015
    lon_min = trace['lon'].min() - margin
    lon_max = trace['lon'].max() + margin
    lat_min = trace['lat'].min() - margin
    lat_max = trace['lat'].max() + margin

    fig, ax = plt.subplots(figsize=(10, 8))
    if use_basemap and _add_basemap(ax, lon_min, lon_max, lat_min, lat_max):
        pass
    else:
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    n_veh = trace['vehicle_id'].nunique()
    try:
        cmap = plt.colormaps['tab10'].resampled(max(n_veh, 10))
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap('tab10', max(n_veh, 10))

    # Station markers (fixed)
    stn_lats = [s['lat'] for s in stations]
    stn_lons = [s['lon'] for s in stations]
    ax.scatter(stn_lons, stn_lats, c='green', s=120, marker='^', edgecolors='darkgreen',
               linewidths=1, label='Stations', zorder=5)

    # Trail history (so far) and current positions
    trail_lines = {}
    for i, vid in enumerate(sorted(trace['vehicle_id'].unique())):
        trail_lines[vid], = ax.plot([], [], color=cmap(i % 10), alpha=0.6, linewidth=1)
    scatter_pts = ax.scatter([], [], c=[], cmap=cmap, s=60, edgecolors='k', linewidths=0.5, zorder=6, vmin=0, vmax=9)

    time_text = ax.set_title('')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    def init():
        for vid in trail_lines:
            trail_lines[vid].set_data([], [])
        scatter_pts.set_offsets(np.empty((0, 2)))
        return list(trail_lines.values()) + [scatter_pts]

    def update(frame_idx):
        t = times[frame_idx]
        sub = trace[trace['time_sec'] <= t]
        for vid in trail_lines:
            vsub = sub[sub['vehicle_id'] == vid].sort_values('time_sec')
            trail_lines[vid].set_data(vsub['lon'].tolist(), vsub['lat'].tolist())
        current = trace[trace['time_sec'] == t]
        if len(current) > 0:
            scatter_pts.set_offsets(current[['lon', 'lat']].values)
            scatter_pts.set_array(current['vehicle_id'].values)
        # Time label (convert sec to HH:MM)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        time_text.set_text(f'{title} — {h:02d}:{m:02d}')
        return list(trail_lines.values()) + [scatter_pts]

    anim = FuncAnimation(fig, update, init_func=init, frames=len(times), interval=80, blit=False)
    if output_path.suffix.lower() == '.gif':
        writer = PillowWriter(fps=10)
        anim.save(str(output_path), writer=writer)
        print(f"Saved animation to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize vehicle movement from trace CSV')
    parser.add_argument('trace_csv', type=str, nargs='?',
                        default=None,
                        help='Path to *_vehicle_trace.csv (e.g. ../03_MILP/milp_vehicle_trace.csv)')
    parser.add_argument('--animate', action='store_true', help='Produce time animation (GIF)')
    parser.add_argument('--no-basemap', action='store_true',
                        help='Do not add OSM basemap (roads/buildings). Use if contextily missing or no network.')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file (e.g. movement.png or movement.gif)')
    parser.add_argument('--title', type=str, default=None, help='Plot title')
    args = parser.parse_args()
    use_basemap = not args.no_basemap

    # Default trace path: 03_MILP relative to this script, or cwd
    here = Path(__file__).resolve().parent
    if args.trace_csv:
        trace_path = Path(args.trace_csv)
    else:
        candidates = [
            here / 'milp_vehicle_trace.csv',
            here / '03_MILP' / 'milp_vehicle_trace.csv',
            here.parent / '03_MILP' / 'milp_vehicle_trace.csv',
            Path('03_MILP/milp_vehicle_trace.csv'),
            Path('milp_vehicle_trace.csv'),
        ]
        trace_path = next((p for p in candidates if p.exists()), None)
        if trace_path is None:
            print("No trace CSV given and none found. Usage:")
            print("  python visualize_movement.py <path_to_vehicle_trace.csv> [--animate] [--output out.png]")
            print("Example: python visualize_movement.py ../03_MILP/milp_vehicle_trace.csv")
            return 1

    if not trace_path.exists():
        print(f"File not found: {trace_path}")
        return 1

    trace = load_trace(trace_path)
    title = args.title or f"Vehicle movement — {trace_path.stem.replace('_vehicle_trace', '')}"

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib")
        return 1

    if args.animate:
        out = Path(args.output) if args.output else here / 'movement.gif'
        plot_animated(trace, STATIONS, out, title=title, use_basemap=use_basemap)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_static(trace, STATIONS, ax, title=title, use_basemap=use_basemap)
        out = Path(args.output) if args.output else here / 'movement.png'
        fig.tight_layout()
        fig.savefig(out, dpi=120, bbox_inches='tight')
        print(f"Saved static plot to {out}")
        plt.close(fig)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
