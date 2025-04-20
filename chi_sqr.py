import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
columns = ["Accident_Severity", "Day_of_Week", "Road_Type", "Junction_Detail", "Junction_Control", "Ped_Cross-Human_Control", "Ped_Cross-Physical_Facil", "Light_Conditions", "Weather_Conditions", "Road_Surface_Conditions", "Special_Conditions_at_Site", "Carriageway_Hazards", "Urban_or_Rural_Area", "Police_Officer_Attend"]
target_column = "Accident_Severity"
for i in range (1, 5) :
    df = pd.read_csv (f"Clusters\\cluster_{i}.csv", usecols = columns)
    df = df.replace (-1, np.nan) ; df = df.dropna ()
    print (f"Cluster {i} :", len (df[df[target_column] == 1]), len (df[df[target_column] == 2]), len (df[df[target_column] == 3]), )
    y = df[target_column] ; x = df.drop (columns = [target_column])
    chi_scores, p_values = chi2 (x, y)
    results = pd.DataFrame({"Feature" : x.columns, "Chi2_Score" : chi_scores, "P_Value" : p_values}).sort_values (by = "Chi2_Score", ascending = False)
    results = results.head (5)
    print (results) ; print ()