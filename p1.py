import pandas as pd
import numpy as np

stations = ["Aluva", "Edappally", "Maharajas", "Petta"]
dates = pd.date_range("2025-09-01", periods=30, freq='D')

data = []
for date in dates:
    for station in stations:
        entered = np.random.randint(100, 2000)
        exited = np.random.randint(80, 1900)
        fare = entered * np.random.uniform(20, 40)
        temp = np.random.uniform(24, 36)
        rain = np.random.choice([0, np.random.uniform(0, 30)], p=[0.7, 0.3])
        weekday = date.weekday()
        data.append([date, station, entered, exited, fare, temp, rain, weekday])

df = pd.DataFrame(data, columns=["Date", "Station", "Passengers_Entered", "Passengers_Exited", "Fare_Collected", "Temperature", "Rainfall", "Weekday"])