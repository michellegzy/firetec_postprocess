import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 

# read csv
df = pd.read_csv('avg_f1_consumption.csv')

# The first column is the x-axis (time)
x_axis = df.iloc[:, 0]  # 1st col
x_label = df.columns[0]  # col name 

# define color groups using different colormaps for each group
greens = plt.cm.Greens(np.linspace(0.3, 0.9, 5))  # 0.3-0.9 to avoid too light colors
blues = plt.cm.Blues(np.linspace(0.3, 0.9, 5))
reds = plt.cm.Reds(np.linspace(0.3, 0.9, 5))

# combine all colors
colors = np.vstack([greens, blues, reds])
plt.figure(figsize=(10, 6))
for i, col in enumerate(df.columns[1:16]): 
    plt.plot(x_axis, df[col], color=colors[i], label=col)

plt.xlabel(x_label)
plt.ylabel('avg dry fuel consumption rate [kg/m3s]')
plt.title('0% Humidity: Varied Fuel Loading and Moisture, Dry Fuel, Double Fuel Cases')
plt.legend()
plt.tight_layout()

plt.savefig('0hum_f1df_avgfc.png', dpi=300)
plt.show()