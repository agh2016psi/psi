
from random import choice
from pylab import plot, ylim
from numpy import array, dot, random

unit_step = lambda x: 0 if x < 0 else 1


training_data_or = [
    (array([0, 0, 1]), 0),
    (array([0, 1, 1]), 1),
    (array([1, 0, 1]), 1),
    (array([1, 1, 1]), 1),
]

training_data_and = [
    (array([0, 0, 1]), 0),
    (array([0, 1, 1]), 0),
    (array([1, 0, 1]), 0),
    (array([1, 1, 1]), 1),
]

training_data_not = [
    (array([0]), 1),
    (array([1]), 0),
]

"""
Most perception code taken
from https://blog.dbrgn.ch/2013/3/26/perceptrons-in-python/
"""


def solvePerceptron(*training_data1):
    training_data = []
    training_data = training_data1

    w = random.rand(3)
    errors = []
    eta = 0.2
    n = 100

    for i in xrange(n):
        x, expected = choice(training_data)
        result = dot(w, x)
        error = expected - unit_step(result)
        errors.append(error)
        w += eta * error * x

    for x, _ in training_data:
        result = dot(x, w)
        print("{}: {} -> {}".format(x[:2], result, unit_step(result)))

    ylim([-1, 1])
    plot(errors)

solvePerceptron(*training_data_or)
solvePerceptron(*training_data_and)
solvePerceptron(*training_data_not)
