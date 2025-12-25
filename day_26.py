import matplotlib.pyplot as plt
import numpy as np



x = np.array([2015, 2017, 2018])
y= np.array([2,4,6])

plt.plot(x, y, marker = "v", markersize = 15, markerfacecolor = "red")
plt.plot(x, y)

plt.show()