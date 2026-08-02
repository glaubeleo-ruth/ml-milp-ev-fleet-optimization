"""
Side-by-side animation of MILP vs greedy dispatch over one operating day.

Reads both *_vehicle_trace.csv files, interpolates vehicle positions and SOC
between the hourly snapshots so motion is continuous, and renders a GIF with:

  - one panel per policy, sharing a clock and an OSM basemap
  - vehicles colored by state of charge (green = full, red = depleted)
  - fading trails showing recent movement
  - a live cumulative "trips served" counter per policy

Requires: matplotlib, pandas, numpy. For the basemap: pip install contextily
(falls back to plain axes with a grid if contextily or the network is absent).

Usage:
  python animate_comparison.py
  python animate_comparison.py --output ../assets/fleet_comparison.gif
  python animate_comparison.py --frames 90 --fps 12 --no-basemap
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Station positions (same as rolling_horizon / scenario_execution / visualize_movement)
STATIONS = [
    {'name': 'MTCC', 'lat': 41.838385, 'lon': -87.627555},
    {'name': 'Paul Galvin Library', 'lat': 41.833675, 'lon': -87.628336},
    {'name': 'McCormick Student Village', 'lat': 41.835527, 'lon': -87.624207},
    {'name': 'Crown Hall', 'lat': 41.833199, 'lon': -87.627273},
    {'name': 'Kaplan Institute', 'lat': 41.836861, 'lon': -87.628300},
    {'name': 'Arthur S. Keating Sports Center', 'lat': 41.838985, 'lon': -87.625566},
]

TRAIL_SECONDS = 2400  # how much history each trail keeps (40 min)


def load_trace(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ('time_sec', 'vehicle_id', 'lat', 'lon')
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s): {missing}")
    return df.sort_values(['vehicle_id', 'time_sec']).reset_index(drop=True)


def interpolate(trace: pd.DataFrame, grid: np.ndarray) -> dict:
    """Resample each vehicle onto a common time grid.

    The raw trace is hourly, so vehicles teleport between stations. Linear
    interpolation turns each hop into continuous travel along a straight line,
    which is what makes the animation read as movement rather than blinking.
    """
    out = {}
    for vid, sub in trace.groupby('vehicle_id'):
        sub = sub.sort_values('time_sec')
        t = sub['time_sec'].to_numpy()
        series = {
            'lat': np.interp(grid, t, sub['lat'].to_numpy()),
            'lon': np.interp(grid, t, sub['lon'].to_numpy()),
        }
        for col in ('soc', 'trips_served'):
            if col in sub.columns:
                series[col] = np.interp(grid, t, sub[col].to_numpy())
        out[vid] = series
    return out


def add_basemap(ax, extent, zoom=17) -> bool:
    """Draw an OSM basemap. Returns False if contextily or the network is unavailable."""
    lon_min, lon_max, lat_min, lat_max = extent
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    try:
        import contextily as ctx
        ctx.add_basemap(ax, crs='EPSG:4326', zoom=zoom,
                        source=ctx.providers.CartoDB.Positron, attribution_size=5)
        return True
    except Exception:
        return False


def build(traces: dict, output: Path, frames: int, fps: int, use_basemap: bool, dpi: int):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    labels = list(traces.keys())
    all_rows = pd.concat(traces.values())

    t_min = max(df['time_sec'].min() for df in traces.values())
    t_max = min(df['time_sec'].max() for df in traces.values())
    grid = np.linspace(t_min, t_max, frames)
    interp = {name: interpolate(df, grid) for name, df in traces.items()}

    margin = 0.0008
    extent = (
        all_rows['lon'].min() - margin, all_rows['lon'].max() + margin,
        all_rows['lat'].min() - margin, all_rows['lat'].max() + margin,
    )

    # Green (full) through amber to red (depleted) — makes the greedy stall legible.
    soc_cmap = LinearSegmentedColormap.from_list(
        'soc', ['#c0392b', '#e67e22', '#f1c40f', '#27ae60'])
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(1, len(labels), figsize=(12.5, 6.4))
    if len(labels) == 1:
        axes = [axes]
    fig.patch.set_facecolor('white')

    artists = {}
    for ax, name in zip(axes, labels):
        drew = use_basemap and add_basemap(ax, extent)
        if not drew:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#d0d0d0')

        ax.scatter([s['lon'] for s in STATIONS], [s['lat'] for s in STATIONS],
                   marker='^', s=110, c='#1a5276', edgecolors='white',
                   linewidths=1.2, zorder=4, label='Charging station')

        vids = sorted(interp[name].keys())
        trails = {vid: ax.plot([], [], color='#34495e', alpha=0.35,
                               linewidth=1.1, zorder=5)[0] for vid in vids}
        dots = ax.scatter(np.zeros(len(vids)), np.zeros(len(vids)),
                          c=np.ones(len(vids)), cmap=soc_cmap, norm=norm,
                          s=110, edgecolors='white', linewidths=1.0, zorder=6)
        counter = ax.text(0.5, -0.055, '', transform=ax.transAxes, ha='center',
                          va='top', fontsize=12, fontweight='bold', color='#2c3e50')
        ax.set_title(name, fontsize=14, fontweight='bold', pad=10, color='#2c3e50')
        artists[name] = {'trails': trails, 'dots': dots, 'counter': counter, 'vids': vids}

    clock = fig.suptitle('', fontsize=17, fontweight='bold', y=0.965, color='#2c3e50')

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=soc_cmap), ax=axes,
                        fraction=0.02, pad=0.015)
    cbar.set_label('State of charge', fontsize=10)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['0%', '50%', '100%'])
    cbar.ax.tick_params(labelsize=9)

    axes[0].legend(loc='upper left', fontsize=9, framealpha=0.9)

    def update(k):
        t = grid[k]
        changed = []
        for name in labels:
            a = artists[name]
            lats = np.array([interp[name][v]['lat'][k] for v in a['vids']])
            lons = np.array([interp[name][v]['lon'][k] for v in a['vids']])
            socs = np.array([interp[name][v].get('soc', np.ones(frames))[k] for v in a['vids']])

            a['dots'].set_offsets(np.column_stack([lons, lats]))
            a['dots'].set_array(socs)

            lo = max(0, k - int(TRAIL_SECONDS / (grid[1] - grid[0])))
            for v in a['vids']:
                a['trails'][v].set_data(interp[name][v]['lon'][lo:k + 1],
                                        interp[name][v]['lat'][lo:k + 1])
            served = int(sum(interp[name][v].get('trips_served', np.zeros(frames))[k]
                             for v in a['vids']))
            a['counter'].set_text(f'{served:,} trips served')
            changed += list(a['trails'].values()) + [a['dots'], a['counter']]

        h, m = int(t // 3600) % 24, int((t % 3600) // 60)
        clock.set_text(f'EV fleet dispatch — {h:02d}:{m:02d}')
        return changed + [clock]

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // fps, blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output), writer=PillowWriter(fps=fps), dpi=dpi,
              savefig_kwargs={'facecolor': 'white'})
    plt.close(fig)
    print(f"Saved {output} ({output.stat().st_size / 1e6:.1f} MB, {frames} frames)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    p.add_argument('--milp', default=str(here.parent / '03_MILP' / 'milp_vehicle_trace.csv'))
    p.add_argument('--greedy', default=str(here.parent / '03_MILP' / 'greedy_vehicle_trace.csv'))
    p.add_argument('--output', '-o', default=str(here.parent / 'assets' / 'fleet_comparison.gif'))
    p.add_argument('--frames', type=int, default=110)
    p.add_argument('--fps', type=int, default=10)
    p.add_argument('--dpi', type=int, default=80)
    p.add_argument('--no-basemap', action='store_true')
    args = p.parse_args()

    for path in (args.milp, args.greedy):
        if not Path(path).exists():
            print(f"Trace not found: {path}")
            return 1

    traces = {
        'MILP (ML-guided)': load_trace(Path(args.milp)),
        'Greedy baseline': load_trace(Path(args.greedy)),
    }
    build(traces, Path(args.output), args.frames, args.fps,
          not args.no_basemap, args.dpi)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
