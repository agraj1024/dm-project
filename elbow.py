import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from scipy.spatial import cKDTree
from math import radians, cos, sin
def to_cartesian (df) :
    def latlon_to_cartesian (lat, lon) :
        R, lat, lon = 6371, radians (lat), radians (lon)
        return R * cos (lat) * cos (lon), R * cos (lat) * sin (lon), R * sin (lat)
    cartesian_coords = np.array ([latlon_to_cartesian (row['Latitude'], row['Longitude']) for _, row in df.iterrows ()])
    df[['X', 'Y', 'Z']] = cartesian_coords
    return df
def location_index (df, dist_threshold = 1.0) :
    coords = df[['X', 'Y', 'Z']].values
    tree = cKDTree (coords)
    clusters = tree.query_ball_tree (tree, dist_threshold)
    location_indices = np.full (len (df), -1)
    current_index = 1
    for i, neighbors in enumerate (clusters) :
        if location_indices[i] == -1:
            for j in neighbors:
                location_indices[j] = current_index
            current_index += 1
    df['Location_Index'] = location_indices
    return df
def load_data (file_path) :
    df = pd.read_csv (file_path)
    df.replace (-1, np.nan, inplace=True) ; df.dropna (inplace=True)
    df = to_cartesian (df)
    df = location_index (df)
    df_freq = df.groupby (['Location_Index']).agg ({'Latitude': 'first', 'Longitude': 'first', 'X': 'first', 'Y': 'first', 'Z': 'first', 'Location_Index': 'count'}).rename (columns = {'Location_Index': 'Frequency'}).reset_index ()
    print (len (df_freq), df_freq['Frequency'].min (), df_freq['Frequency'].max ())
    Q1_freq = df_freq['Frequency'].quantile (0.25)
    Q3_freq = df_freq['Frequency'].quantile (0.75)
    IQR_freq = Q3_freq - Q1_freq
    lower_bound = Q1_freq - IQR_freq
    upper_bound = Q3_freq + IQR_freq
    print (upper_bound, lower_bound, len (df_freq[df_freq['Frequency'] > upper_bound]))
    df_freq = df_freq[(df_freq['Frequency'] >= lower_bound) & (df_freq['Frequency'] <= upper_bound)]
    print (len (df_freq))
    df = df[df['Location_Index'].isin (df_freq['Location_Index'])]
    print (len (df))
    df_freq['Frequency'] = np.sqrt (df_freq['Frequency'])
    df_freq[['X', 'Y', 'Z']] *= df_freq['Frequency'].values[:, np.newaxis]
    return df, df_freq
def plot_accident_frequency (data) :
    data_sorted = data.sort_values (by = 'Frequency')
    plt.figure (figsize= (10, 6))
    plt.plot (range (len (data_sorted)), data_sorted['Frequency'], marker = 'o', linestyle = '-', color = 'b')
    plt.xlabel ("Location Index")
    plt.ylabel ("Accident Frequency")
    plt.title ("Accident Frequency Distribution Across Indexed Locations")
    plt.show ()
def gap_statistics (data, max_k = 10) :
    Wk_values = []
    k_values = list (range (1, max_k + 1))
    data_cartesian = data[['X', 'Y', 'Z']].values
    for k in k_values :
        kmeans = KMeans (n_clusters = k, random_state = 42).fit (data_cartesian)
        Wk = sum (np.min (kmeans.transform (data_cartesian), axis = 1))
        Wk_values.append (Wk)
    plt.figure (figsize = (8, 5))
    plt.plot (k_values, Wk_values, marker='o', linestyle='-', label = 'Within-Cluster Sum of Squares (Wk)')
    plt.xlabel ('Number of Clusters (k)')
    plt.ylabel ('Within-Group Sum of Squares')
    plt.title ('Gap Statistics for Optimal k')
    plt.legend ()
    plt.show ()
file_path = "Dataset/Accidents.csv"
df, df_freq = load_data (file_path)
plot_accident_frequency (df_freq)
gap_statistics (df_freq)