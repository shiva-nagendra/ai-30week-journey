#gradient_descent from scratch

import random

def f(x):
    return x**2
def df(x):
    return x*2

x = random.uniform(-5, 5)
learning_rate = 0.1
steps = 20

print(f"starting x:{x}\n")

for i in range(steps):
    grad = df(x)
    x = x - learning_rate*grad
    y = f(x)

    print(f"step {i+1}: x = {x:.3f}, y = {y:.3f}")