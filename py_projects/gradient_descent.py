#gradient_descent from scratch

#gradient_descent from scratch

import random
import matplotlib.pyplot as plt


def f(x):
    return x**2
def df(x):
    return x*2

x = random.uniform(-5, 5)
learning_rate = 0.1
steps = 20

print(f"starting x:{x}\n")

xs = []
ys = []

for i in range(steps):
    grad = df(x)
    x = x - learning_rate*grad
    y = f(x)

    xs.append(x)
    ys.append(y)

    print(f"step {i+1}: x = {x:.3f}, y = {y:.3f}")

#plotting parabola   
import numpy as np
x_curve = np.linspace(-6,6,100)
y_curve = f(x_curve)
plt.plot(x_curve, y_curve)

#plot gradient decent path
plt.scatter(xs,ys)
plt.plot(xs,ys)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Gradient decent on y = x**2")
plt.grid(True)

plt.show()