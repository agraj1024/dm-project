import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
for c in [1, 2, 3, 4] :
    df = pd.read_csv (f'Clusters\\cluster_{c}.csv', usecols = ["Date"])
    df['Year'] = pd.to_datetime(df['Date'], format='%d/%m/%Y').dt.year
    actual = len (df[df['Year'] == 2014])
    df = df[df['Year'] < 2014]
    time_series = df.groupby ('Year').size ()
    print (f"Fitting ARIMA model for cluster {c}")
    model = ARIMA (time_series, order = (1, 0, 1))
    model_fit = model.fit ()
    forecast = model_fit.forecast (steps = 1)
    plt.figure (figsize = (8, 4))
    plt.plot (time_series, label = "Actual Data", marker = 'o')
    plt.axvline (x = 2013, color = 'r', linestyle = '--', label = "Forecast Start")
    plt.scatter (2014, forecast.iloc[0], color = 'red', marker = 'x', s = 100, label = "Predicted (2014)")
    plt.xlabel ("Year")
    plt.ylabel ("Accident Count")
    plt.title (f"2014 accident forecast for cluster {c}")
    plt.legend ()
    plt.show ()
    prediction = forecast.iloc[0]
    print (f"Predicted accidents for cluster {c} : {prediction:.2f}\nActual accidents : {actual}\n")