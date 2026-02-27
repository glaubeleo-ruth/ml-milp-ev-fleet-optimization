import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# STEP 1: Define All Buildings with Coordinates and Categories
# =============================================================================

# All 40 IIT campus buildings with real coordinates from OpenStreetMap
buildings_data = {
    'Perlstein Hall': {'coords': (41.835479, -87.627146), 'category': 'Academic', 'priority': 1},
    'McCormick Tribune Campus Center': {'coords': (41.835621, -87.625887), 'category': 'Student Center', 'priority': 1},
    'Pi Kappa Phi': {'coords': (41.833902, -87.624635), 'category': 'Greek Housing', 'priority': 4},
    'Alpha Sigma Alpha Sorority': {'coords': (41.833645, -87.623749), 'category': 'Greek Housing', 'priority': 4},
    'Alpha Sigma Phi': {'coords': (41.833032, -87.624633), 'category': 'Greek Housing', 'priority': 4},
    'Alumni Memorial Hall': {'coords': (41.836214, -87.627335), 'category': 'Admin', 'priority': 3},
    'Arthur S. Keating Sports Center': {'coords': (41.838985, -87.625566), 'category': 'Recreation', 'priority': 2},
    'Kacek Hall': {'coords': (41.837866, -87.624703), 'category': 'Residence', 'priority': 2},
    'Carmen Hall': {'coords': (41.836690, -87.624691), 'category': 'Residence', 'priority': 2},
    'The Commons': {'coords': (41.836040, -87.625535), 'category': 'Dining', 'priority': 1},
    'Cunningham Hall': {'coords': (41.837740, -87.623975), 'category': 'Residence', 'priority': 2},
    'Delta Tau Delta': {'coords': (41.833323, -87.624644), 'category': 'Greek Housing', 'priority': 4},
    'Rettaliata Engineering Center': {'coords': (41.837159, -87.627185), 'category': 'Academic', 'priority': 1},
    'Farr Hall': {'coords': (41.834344, -87.623795), 'category': 'Residence', 'priority': 2},
    'Paul Galvin Library': {'coords': (41.833675, -87.628336), 'category': 'Library', 'priority': 1},
    'Grover M. Herman Hall': {'coords': (41.835681, -87.628387), 'category': 'Admin', 'priority': 3},
    'Gunsaulus Hall': {'coords': (41.837066, -87.623934), 'category': 'Residence', 'priority': 2},
    'Michael Galvin Tower': {'coords': (41.831394, -87.627231), 'category': 'Residence', 'priority': 2},
    'Kappa Phi Delta': {'coords': (41.833919, -87.623753), 'category': 'Greek Housing', 'priority': 4},
    'Pritzker Science Center': {'coords': (41.837930, -87.627380), 'category': 'Academic', 'priority': 1},
    'Machinery Hall': {'coords': (41.834819, -87.629215), 'category': 'Academic', 'priority': 2},
    'Main Building': {'coords': (41.834303, -87.629281), 'category': 'Academic', 'priority': 1},
    'Materials Technology Building': {'coords': (41.833236, -87.629243), 'category': 'Academic', 'priority': 2},
    'McCormick Student Village': {'coords': (41.835527, -87.624207), 'category': 'Residence', 'priority': 1},
    'Phi Kappa Sigma': {'coords': (41.833041, -87.624172), 'category': 'Greek Housing', 'priority': 4},
    'S.R. Crown Hall': {'coords': (41.833199, -87.627273), 'category': 'Academic', 'priority': 1},
    'Siegel Hall': {'coords': (41.834250, -87.627603), 'category': 'Academic', 'priority': 1},
    'Sigma Phi Epsilon': {'coords': (41.833604, -87.624624), 'category': 'Greek Housing', 'priority': 4},
    'Rowe Village Middle': {'coords': (41.833712, -87.626167), 'category': 'Residence', 'priority': 2},
    'Stuart Hall': {'coords': (41.838746, -87.627396), 'category': 'Academic', 'priority': 1},
    'Triangle Fraternity': {'coords': (41.833038, -87.623692), 'category': 'Greek Housing', 'priority': 4},
    'Vandercook College Of Music': {'coords': (41.836762, -87.629165), 'category': 'Academic', 'priority': 2},
    'Wishnick Hall': {'coords': (41.835094, -87.627614), 'category': 'Academic', 'priority': 1},
    'Carr Memorial Chapel': {'coords': (41.836260, -87.624388), 'category': 'Other', 'priority': 4},
    'IIT (Automotive Lab)': {'coords': (41.835178, -87.629208), 'category': 'Academic', 'priority': 3},
    'Tech South': {'coords': (41.831730, -87.627238), 'category': 'Academic', 'priority': 2},
    'Tech North': {'coords': (41.832509, -87.627287), 'category': 'Academic', 'priority': 2},
    'Tech Center': {'coords': (41.832153, -87.627251), 'category': 'Academic', 'priority': 2},
    'Chemistry Research Building': {'coords': (41.831886, -87.628445), 'category': 'Academic', 'priority': 2},
    'Kaplan Institute': {'coords': (41.836861, -87.628300), 'category': 'Academic', 'priority': 1},
}

# Convert to DataFrame
df = pd.DataFrame([
    {
        'name': name,
        'lat': data['coords'][0],
        'lon': data['coords'][1],
        'category': data['category'],
        'priority': data['priority']
    }
    for name, data in buildings_data.items()
])

print(f"Total buildings: {len(df)}")
print(f"\nCategory distribution:")
print(df['category'].value_counts().to_string())


# =============================================================================
# STEP 2: K-Means Clustering (K=15)
# =============================================================================

N_CLUSTERS = 15

# Standardize coordinates for clustering
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(df[['lat', 'lon']].values)

# Run K-Means
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(coords_scaled)

print(f"\nK-Means clustering with K={N_CLUSTERS}")
print(f"Buildings per cluster:")
print(df.groupby('cluster').size().to_string())


# =============================================================================
# STEP 3: Select Best Building from Each Cluster
# =============================================================================

selected = []

for cluster_id in range(N_CLUSTERS):
    cluster_buildings = df[df['cluster'] == cluster_id].copy()
    
    # Sort by priority (1 = highest), then alphabetically for ties
    cluster_buildings = cluster_buildings.sort_values(
        ['priority', 'name'], ascending=[True, True]
    )
    
    # Select highest-priority building
    best = cluster_buildings.iloc[0]
    selected.append(best)
    
    print(f"  Cluster {cluster_id}: {best['name']} "
          f"(priority={best['priority']}, category={best['category']})")

selected_df = pd.DataFrame(selected).reset_index(drop=True)


# =============================================================================
# STEP 4: Category Coverage Check
# =============================================================================

# Essential categories that must be represented
essential_categories = [
    'Academic', 'Residence', 'Library', 'Student Center',
    'Dining', 'Recreation'
]

print(f"\nSelected POIs categories: {selected_df['category'].unique().tolist()}")

# Check for missing essential categories
missing = [cat for cat in essential_categories
           if cat not in selected_df['category'].values]

if missing:
    print(f"Missing essential categories: {missing}")
    
    # Add highest-priority building from each missing category
    for cat in missing:
        candidates = df[
            (df['category'] == cat) &
            (~df['name'].isin(selected_df['name']))
        ].sort_values('priority')
        
        if len(candidates) > 0:
            addition = candidates.iloc[0]
            selected_df = pd.concat(
                [selected_df, pd.DataFrame([addition])],
                ignore_index=True
            )
            print(f"  Added: {addition['name']} ({cat})")
else:
    print("All essential categories covered!")

print(f"\nFinal selection: {len(selected_df)} POIs")


# =============================================================================
# STEP 5: Export Selected POIs
# =============================================================================

export_df = selected_df[['name', 'lat', 'lon', 'category', 'priority']].copy()
export_df.columns = ['poi_name', 'latitude', 'longitude', 'category', 'priority']
export_df = export_df.sort_values('poi_name').reset_index(drop=True)

export_df.to_csv('selected_pois.csv', index=False)
print("\nSaved: selected_pois.csv")


# =============================================================================
# STEP 6: Summary
# =============================================================================

print("\n" + "=" * 60)
print("FINAL SELECTED POIs")
print("=" * 60)

for _, row in export_df.iterrows():
    print(f"  {row['poi_name']:40s} | {row['category']:15s} | priority={row['priority']}")

print(f"\n  Total: {len(export_df)} POIs")
print(f"\n  Category distribution:")
for cat, count in export_df['category'].value_counts().items():
    print(f"    {cat}: {count}")


if __name__ == "__main__":
    pass