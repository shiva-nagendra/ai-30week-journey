#Linear algebra visualisation

import matplotlib.pyplot as plt

# original vector
v = [-1, 3]

plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.grid()
plt.title("Original Vector")
plt.show()