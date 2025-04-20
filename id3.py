import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
columns = ["Accident_Severity", "Day_of_Week", "Road_Type", "Junction_Detail", "Junction_Control", "Ped_Cross-Human_Control", "Ped_Cross-Physical_Facil", "Light_Conditions", "Weather_Conditions", "Road_Surface_Conditions", "Special_Conditions_at_Site", "Carriageway_Hazards", "Urban_or_Rural_Area", "Police_Officer_Attend"]
target_column = "Accident_Severity"
mapping_data = pd.read_excel ("Dataset\\Lookup.xls")
for i in range (1, 5) :
    print (f"Cluster {i} :")
    df = pd.read_csv (f"Clusters\\cluster_{i}.csv", usecols = columns)
    df.replace (-1, np.nan, inplace = True) ; df.dropna (inplace = True)
    y = df[target_column] ; x = df.drop (columns = [target_column])
    chi_scores, p_values = chi2 (x, y)
    results = pd.DataFrame({"Feature" : x.columns, "Chi2_Score" : chi_scores, "P_Value" : p_values}).sort_values (by = "Chi2_Score", ascending = False)
    results = results.head (5)
    x = x.drop (columns = [col for col in x.columns if col not in results["Feature"].tolist ()])
    x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.1, random_state = 42)
    clf = DecisionTreeClassifier (criterion='entropy', random_state = 42)
    clf.fit (x_train, y_train)
    y_pred = clf.predict (x_test)
    acc = accuracy_score (y_test, y_pred)
    print (f"Accuracy: {acc:.4f}")
    print ("Classification Report :")
    print (classification_report (y_test, y_pred))
    plt.figure(figsize=(14, 6))
    plot_tree (clf, feature_names = x.columns, class_names = [str(c) for c in clf.classes_], filled = True, max_depth = 3, fontsize = 8)
    plt.title (f"Decision Tree for cluster {i}")
    plt.show()
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(f'Confusion Matrix for cluster {i}')
    plt.show()