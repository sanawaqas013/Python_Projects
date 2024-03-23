import numpy as np
import matplotlib.pyplot as plt

X0 = 5
M = 2
theta = 0.5
generations = 10

def branching_process(X0, M, theta, generations):
    X = np.zeros(generations, dtype=int)
    X[0] = X0

    for t in range(1, generations):
        infections = np.random.negative_binomial(n=theta, p=theta / (theta + M), size=X[t - 1])
        X[t] = np.sum(infections)
    return X

Infected = branching_process(X0, M, theta, generations)

def analytical_extinction_prob(M, theta):
    p1 = (theta / (M + theta)) ** theta
    return p1 / (1 - (1 / (M + theta)))

def extinction_prob(X):
    extinction_probability = (np.sum(X == 0)) / (len(X))
    return extinction_probability

print(Infected)
print("Analytical Extinction Probability:", analytical_extinction_prob (M, theta))
print("Extinction Probability:", extinction_prob(Infected))

def growth_rate(X):
    rate = np.mean(X[1:] / X[:-1])
    return rate

print("Growth rate:", growth_rate(Infected))
expected_growth_rate = M / theta
print("Expected growth rate:", expected_growth_rate)

#plt.plot(range(generations), Infected, marker='o')
#plt.xlabel("Generation")
#plt.ylabel("Number of Infected Individuals")
#plt.title("Branching Process Simulation")
#plt.show()