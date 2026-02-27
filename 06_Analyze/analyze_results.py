import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List
import argparse


def load_results(results_dir: Path) -> tuple:
    """Load scenario results JSON and summary CSV."""
    results_json = results_dir / 'scenario_results.json'
    summary_csv = results_dir / 'scenario_summary.csv'
    
    results = None
    df = None
    
    if results_json.exists():
        with open(results_json) as f:
            results = json.load(f)
    
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
    
    return results, df


def plot_service_rate_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Bar chart comparing service rates across methods and scenarios.
    """
    if 'temperature' not in df.columns:
        print("Skipping service rate plot: no temperature column")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Pivot for grouped bars (aggregate so index has no duplicates)
    pivot = df.pivot_table(
        index='temperature',
        columns='method',
        values='service_rate_pct',
        aggfunc='mean'
    )
    pivot = pivot.loc[~pivot.index.duplicated(keep='first')]

    # Reorder (use unique target order to avoid reindex duplicate-label error)
    temp_order = ['cold_winter', 'baseline', 'hot_summer']
    order_list = [t for t in temp_order if t in pivot.index]
    pivot = pivot.reindex(order_list)
    method_order = ['milp_ml', 'milp_fixed', 'nearest']
    pivot = pivot[[m for m in method_order if m in pivot.columns]]
    
    # Colors
    colors = {'milp_ml': '#2ecc71', 'milp_fixed': '#3498db', 'nearest': '#e74c3c'}
    method_labels = {'milp_ml': 'MILP + ML', 'milp_fixed': 'MILP + Fixed', 'nearest': 'Nearest-Available'}
    
    # Plot
    x = np.arange(len(pivot.index))
    width = 0.25
    
    for i, method in enumerate(pivot.columns):
        bars = ax.bar(
            x + i * width, 
            pivot[method], 
            width, 
            label=method_labels.get(method, method),
            color=colors.get(method, '#95a5a6')
        )
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Temperature Scenario')
    ax.set_ylabel('Service Rate (%)')
    ax.set_title('Service Rate Comparison by Temperature and Method')
    ax.set_xticks(x + width)
    ax.set_xticklabels([t.replace('_', '\n') for t in pivot.index])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'service_rate_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'service_rate_comparison.png'}")


def plot_energy_by_temperature(df: pd.DataFrame, output_dir: Path):
    """
    Line/bar chart showing energy consumption by temperature scenario.
    Highlights where ML prediction differs from fixed rate.
    """
    if 'temperature' not in df.columns:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Total energy by method
    ax1 = axes[0]
    pivot_energy = df.pivot_table(
        index='temperature',
        columns='method',
        values='total_energy_kWh',
        aggfunc='mean'
    )
    pivot_energy = pivot_energy.loc[~pivot_energy.index.duplicated(keep='first')]

    temp_order = ['cold_winter', 'baseline', 'hot_summer']
    order_list = [t for t in temp_order if t in pivot_energy.index]
    pivot_energy = pivot_energy.reindex(order_list)
    
    colors = {'milp_ml': '#2ecc71', 'milp_fixed': '#3498db', 'nearest': '#e74c3c'}
    
    pivot_energy.plot(kind='bar', ax=ax1, color=[colors.get(c, '#95a5a6') for c in pivot_energy.columns])
    ax1.set_xlabel('Temperature Scenario')
    ax1.set_ylabel('Total Energy (kWh)')
    ax1.set_title('Total Energy Consumption by Method')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title='Method')
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: ML vs Fixed difference
    ax2 = axes[1]
    
    # One value per temperature (avoid duplicate index for reindex)
    ml_data = df[df['method'] == 'milp_ml'].groupby('temperature')['total_energy_kWh'].mean()
    fixed_data = df[df['method'] == 'milp_fixed'].groupby('temperature')['total_energy_kWh'].mean()
    ml_data = ml_data.loc[~ml_data.index.duplicated(keep='first')]
    fixed_data = fixed_data.loc[~fixed_data.index.duplicated(keep='first')]

    # Calculate percentage difference
    order_list = [t for t in temp_order if t in ml_data.index]
    pct_diff = ((fixed_data - ml_data) / fixed_data * 100).reindex(order_list)
    
    colors_bar = ['#e74c3c' if x < 0 else '#2ecc71' for x in pct_diff]
    bars = ax2.bar(pct_diff.index, pct_diff, color=colors_bar)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Temperature Scenario')
    ax2.set_ylabel('Energy Difference (%)')
    ax2.set_title('ML Energy Prediction vs Fixed Rate\n(positive = ML saves energy)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, pct_diff):
        height = bar.get_height()
        ax2.annotate(f'{val:+.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords='offset points',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'energy_by_temperature.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'energy_by_temperature.png'}")


def plot_fleet_size_sensitivity(df: pd.DataFrame, output_dir: Path):
    """
    Line chart showing service rate vs fleet size.
    """
    if 'fleet_size' not in df.columns or df['fleet_size'].nunique() < 2:
        print("Skipping fleet size plot: insufficient variation")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'milp_ml': '#2ecc71', 'milp_fixed': '#3498db', 'nearest': '#e74c3c'}
    markers = {'milp_ml': 'o', 'milp_fixed': 's', 'nearest': '^'}
    labels = {'milp_ml': 'MILP + ML', 'milp_fixed': 'MILP + Fixed', 'nearest': 'Nearest-Available'}
    
    for method in ['milp_ml', 'milp_fixed', 'nearest']:
        method_data = df[df['method'] == method].groupby('fleet_size')['service_rate_pct'].mean()
        ax.plot(method_data.index, method_data.values, 
                marker=markers.get(method, 'o'),
                color=colors.get(method, '#95a5a6'),
                label=labels.get(method, method),
                linewidth=2, markersize=8)
    
    ax.set_xlabel('Fleet Size (vehicles)')
    ax.set_ylabel('Service Rate (%)')
    ax.set_title('Service Rate vs Fleet Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fleet_size_sensitivity.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'fleet_size_sensitivity.png'}")


def plot_ml_contribution_analysis(results: List[Dict], output_dir: Path):
    """
    Create summary visualization of ML's contribution.
    """
    if not results:
        return
    
    # Extract ML vs Fixed deltas
    data = []
    for r in results:
        cfg = r['config']
        deltas = r.get('deltas', {}).get('ml_vs_fixed', {})
        if deltas:
            data.append({
                'scenario': cfg['scenario_id'],
                'temperature': cfg.get('temperature_name', 'unknown'),
                'fleet_size': cfg.get('fleet_size', 10),
                'trips_diff': deltas.get('trips_diff', 0),
                'energy_pct': deltas.get('energy_pct_reduction', 0),
            })
    
    if not data:
        return
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by temperature
    temp_colors = {
        'cold_winter': '#3498db',
        'baseline': '#95a5a6', 
        'hot_summer': '#e74c3c'
    }
    
    colors = [temp_colors.get(t, '#95a5a6') for t in df['temperature']]
    
    scatter = ax.scatter(df['trips_diff'], df['energy_pct'], 
                        c=colors, s=100, alpha=0.7, edgecolors='white')
    
    # Add quadrant labels
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel('Trip Difference (ML - Fixed)')
    ax.set_ylabel('Energy Reduction (%)\n(positive = ML saves energy)')
    ax.set_title('ML Contribution: Service vs Energy Trade-off')
    
    # Legend
    handles = [mpatches.Patch(color=c, label=t.replace('_', ' ').title()) 
               for t, c in temp_colors.items()]
    ax.legend(handles=handles, title='Temperature')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ml_contribution_analysis.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'ml_contribution_analysis.png'}")


def generate_summary_table(df: pd.DataFrame, results: List[Dict], output_dir: Path):
    """
    Generate HTML summary table.
    """
    html = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
            th { background-color: #3498db; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .highlight { background-color: #d5f5e3; font-weight: bold; }
            .negative { color: #e74c3c; }
            .positive { color: #27ae60; }
        </style>
    </head>
    <body>
        <h1>Milestone 3.4 — Scenario Execution Results</h1>
    """
    
    if df is not None and len(df) > 0:
        html += "<h2>Service Rate Summary</h2>"
        
        # Pivot table
        if 'temperature' in df.columns:
            pivot = df.pivot_table(
                index=['temperature', 'fleet_size'],
                columns='method',
                values='service_rate_pct',
                aggfunc='mean'
            ).round(1)
            pivot = pivot.loc[~pivot.index.duplicated(keep='first')]
            html += pivot.to_html(classes='summary-table')
        
        # Raw data
        html += "<h2>Detailed Results</h2>"
        html += df.round(2).to_html(classes='detailed-table', index=False)
    
    if results:
        html += "<h2>ML vs Fixed-Rate Comparison</h2>"
        html += "<table>"
        html += "<tr><th>Scenario</th><th>Temperature</th><th>Fleet</th>"
        html += "<th>Trips Diff</th><th>Energy Diff (%)</th></tr>"
        
        for r in results:
            cfg = r['config']
            deltas = r.get('deltas', {}).get('ml_vs_fixed', {})
            if deltas:
                trips_diff = deltas.get('trips_diff', 0)
                energy_pct = deltas.get('energy_pct_reduction', 0)
                
                trips_class = 'positive' if trips_diff > 0 else ('negative' if trips_diff < 0 else '')
                energy_class = 'positive' if energy_pct > 0 else ('negative' if energy_pct < 0 else '')
                
                html += f"<tr>"
                html += f"<td>{cfg['scenario_id']}</td>"
                html += f"<td>{cfg.get('temperature_name', 'N/A')}</td>"
                html += f"<td>{cfg.get('fleet_size', 'N/A')}</td>"
                html += f"<td class='{trips_class}'>{trips_diff:+d}</td>"
                html += f"<td class='{energy_class}'>{energy_pct:+.1f}%</td>"
                html += f"</tr>"
        
        html += "</table>"
    
    html += """
    </body>
    </html>
    """
    
    path = output_dir / 'scenario_summary.html'
    with open(path, 'w') as f:
        f.write(html)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze scenario results')
    parser.add_argument('--results-dir', type=str, default='scenario_outputs',
                        help='Directory containing scenario_results.json and scenario_summary.csv (default: scenario_outputs)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: same as results)')
    
    args = parser.parse_args()
    _here = Path(__file__).resolve().parent
    _project_root = _here.parent

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        # Resolve relative path: try cwd, then script dir, then project root
        candidates = [
            results_dir,
            Path.cwd() / results_dir,
            _here / results_dir,
            _project_root / str(results_dir),
            _project_root / 'scenario_outputs',
            _project_root / '05_Scenario' / 'scenario_outputs',
            Path('scenario_outputs'),
        ]
        for candidate in candidates:
            if (candidate / 'scenario_summary.csv').exists() or (candidate / 'scenario_results.json').exists():
                results_dir = candidate.resolve()
                break
        else:
            results_dir = (Path.cwd() / results_dir).resolve()
    else:
        results_dir = results_dir.resolve()

    output_dir = Path(args.output_dir).resolve() if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {results_dir}")
    results, df = load_results(results_dir)
    
    if df is None or len(df) == 0:
        print("No summary CSV found. Run scenario_execution.py first (e.g. python 05_Scenario/scenario_execution.py --key).")
        return
    
    print(f"Loaded {len(df)} result rows")
    print(f"Generating visualizations in: {output_dir}")
    
    # Generate plots
    plot_service_rate_comparison(df, output_dir)
    plot_energy_by_temperature(df, output_dir)
    plot_fleet_size_sensitivity(df, output_dir)
    
    if results:
        plot_ml_contribution_analysis(results, output_dir)
    
    generate_summary_table(df, results, output_dir)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()