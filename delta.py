
import numpy as np


def choice(x):
    if(x > 0):
        return 1
    else:
        return 0

# input and output dataset
training_data = [
    (np.array([0, 0, 1]), 0),
    (np.array([0, 1, 1]), 1),
    (np.array([1, 0, 1]), 1),
    (np.array([1, 1, 1]), 1),
]

w = random.rand(3)
a = []
mu = 0.001
n = 50
for i in xrange(n):
    x, expected = choice(training_data)
    result = dot(w, x)
    delta = abs(expected - result)
    a.append(delta)
    print("delta \n", a)
    if delta < a[-1]:
        w += mu * delta * x
    else:
        w -= mu * delta * x
plt.plot(a)
