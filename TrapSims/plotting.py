import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

f = pd.read_csv("/Users/jacobbarnhart/Documents/GitHub/trap-sim-tools-python/TrapSims/bladesAndEndcapsData.csv")

f.rename(columns={"0": "x", '1':'y', '2':'z', '3':'E'}, inplace=True)

print(f['z'].mean())

zthr = 1 # Doesn't really matter. Most points are close to z = 0.
ythr = 1e-2
xthr = 1e-2
# Scale of the graph. Increasing this "zooms out" while decreasing it "zooms in".
# A good scale for fourPoints is around .0055. For bladesAndEndcaps, 0.2
scale = .2

# print("max x: ", max(f["x"]))
# print("max y: ", max(f["y"]))
# print("max z: ", max(f["z"]))
# print("max E: ", max(f["E"]))

# print("min x: ", min(f["x"]))
# print("min y: ", min(f["y"]))
# print("min z: ", min(f["z"]))
# print("min E: ", min(f["E"]))

# Select axis
d1 = f.loc[(f['z'] == 0)]

# Create coordinate pairs.
coordinates = list(zip(d1["x"], d1["y"]))

# Setting for consistent axes.
X = np.linspace(-scale, scale)
Y = np.linspace(-scale, scale)

# Convert x, y for interpolation.
X, Y = np.meshgrid(X, Y)
interpolation = interpolate.LinearNDInterpolator(coordinates, d1["E"], fill_value=0)
Z = interpolation(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 500, cmap='viridis')
plt.show()
