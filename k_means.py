import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from math import radians, cos, sin
def to_cartesian (df) :
    def latlon_to_cartesian (lat, lon) :
        R, lat, lon = 6371, radians (lat), radians (lon)
        return R * cos (lat) * cos (lon), R * cos (lat) * sin (lon), R * sin (lat)
    cartesian_coords = np.array ([latlon_to_cartesian (row['Latitude'], row['Longitude']) for _, row in df.iterrows ()])
    df[['X', 'Y', 'Z']] = cartesian_coords
    return df
def assign_location_index(df, threshold=1.0):
    coords = df[['X', 'Y', 'Z']].values
    tree = cKDTree(coords)
    clusters = tree.query_ball_tree(tree, threshold)
    location_indices = np.full(len(df), -1)
    current_index = 1
    for i, neighbors in enumerate(clusters):
        if location_indices[i] == -1:
            for j in neighbors:
                location_indices[j] = current_index
            current_index += 1
    df['Location_Index'] = location_indices
    return df
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.replace (-1, np.nan) ; df = df.dropna()
    df = to_cartesian(df)
    df = assign_location_index(df)
    df_freq = df.groupby(['Location_Index']).agg({'Latitude': 'first', 'Longitude': 'first', 'X': 'first', 'Y': 'first', 'Z': 'first', 'Location_Index': 'count'}).rename(columns={'Location_Index': 'Frequency'}).reset_index()
    df_freq[['X', 'Y', 'Z']] *= df_freq['Frequency'].values[:, np.newaxis]
    Q1_freq = df_freq['Frequency'].quantile(0.25)
    Q3_freq = df_freq['Frequency'].quantile(0.75)
    IQR_freq = Q3_freq - Q1_freq
    lower_bound_freq = Q1_freq - IQR_freq
    upper_bound_freq = Q3_freq + IQR_freq
    df_freq = df_freq[(df_freq['Frequency'] >= lower_bound_freq) & (df_freq['Frequency'] <= upper_bound_freq)]
    df = df[df['Location_Index'].isin(df_freq['Location_Index'])]
    return df, df_freq
def benchmark (df_freq, clusters) :
    print ("Minimum and maximum accident frequency for")
    for cluster in range (clusters) :
        print ("Cluster", cluster + 1, ":", df_freq[df_freq['Cluster'] == cluster]['Frequency'].min (), df_freq[df_freq['Cluster'] == cluster]['Frequency'].max ())
    print ("\nSilhouette Score")
    data_cartesian = df_freq[['X', 'Y', 'Z']].values
    silhouette_values = silhouette_samples(data_cartesian, df_freq['Cluster'])
    df_freq['Silhouette_Score'] = silhouette_values
    overall_silhouette = silhouette_score(data_cartesian, df_freq['Cluster'])
    print(f"Overall : {overall_silhouette:.4f}")
    for cluster in range(clusters):
        cluster_silhouette = df_freq[df_freq['Cluster'] == cluster]['Silhouette_Score'].mean()
        print(f"Cluster {cluster+1} : {cluster_silhouette:.4f}")
    print ("\nDavies-Bouldin Index :", davies_bouldin_score (data_cartesian, df_freq['Cluster']))
def cluster_and_save(df, df_freq, n_clusters):
    data_cartesian = df_freq[['X', 'Y', 'Z']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data_cartesian)
    df_freq['Cluster'] = kmeans.labels_
    benchmark (df_freq, n_clusters)
    # df_freq = df_freq.drop (columns = ['Silhouette_Score'])
    # df = df.merge(df_freq[['Location_Index', 'Cluster']], on='Location_Index', how='left')
    # for cluster in range(n_clusters):
    #     cluster_df = df[df['Cluster'] == cluster].copy()
    #     columns_to_remove = ['Cluster', 'X', 'Y', 'Z', 'Frequency']
    #     cluster_df.drop(columns=[col for col in columns_to_remove if col in cluster_df.columns], inplace=True)
    #     cluster_df.to_csv(f'cluster_{cluster+1}.csv', index=False)
file_path = "Dataset/Accidents.csv"
df, df_freq = load_data(file_path)
cluster_and_save(df, df_freq, 4)