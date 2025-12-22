import random

# define the function (loss)
def f(x):
    return x ** 2

# derivative of the function
def df(x):
    return 2 * x

# start at a random point
x = random.uniform(-10, 10)

learning_rate = 0.1
steps = 20

print(f"Starting x: {x:.4f}\n")

for step in range(steps):
    gradient = df(x)
    x = x - learning_rate * gradient
    y = f(x)

    print(f"Step {step+1:02d} | x = {x:.4f} | y = {y:.4f}")