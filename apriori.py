import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
columns = ["Accident_Severity", "Day_of_Week", "Road_Type", "Junction_Detail", "Junction_Control", "Ped_Cross-Human_Control", "Ped_Cross-Physical_Facil", "Light_Conditions", "Weather_Conditions", "Road_Surface_Conditions", "Special_Conditions_at_Site", "Carriageway_Hazards", "Urban_or_Rural_Area", "Police_Officer_Attend"]
min_support, min_confidence, min_lift = 0.2, 0.5, 1.0
mapping_data = pd.read_excel ("Dataset\\Lookup.xls", sheet_name = None)
for c in [1, 2, 3, 4] :
    df = pd.read_csv (f"Clusters\\cluster_{c}.csv", usecols = columns)
    df = df.replace (-1, np.nan)
    df = df.dropna ()
    for column in df.columns:
        if column in mapping_data:
            mapping_df = mapping_data[column]
            if 'code' in mapping_df.columns and 'label' in mapping_df.columns:
                mapping_dict = dict(zip(mapping_df['code'], mapping_df['label']))
                df[column] = df[column].map(mapping_dict).fillna(df[column])
    transactions = df.apply(lambda row: set(row.astype(str)), axis=1)
    itemset_list = list(transactions)
    unique_items = sorted(set(item for transaction in itemset_list for item in transaction))
    encoded_data = pd.DataFrame(0, index=range(len(itemset_list)), columns=unique_items)
    for i, transaction in enumerate (itemset_list) :
        encoded_data.loc[i, list(transaction)] = 1
    frequent_itemsets = apriori(encoded_data, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules['lift'] >= min_lift]
    print(f"Association Rules for cluster {c}\n")
    rules.to_csv(f"Rules\\cluster{c}.csv", index=False)