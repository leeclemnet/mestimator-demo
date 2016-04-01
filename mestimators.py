import numpy as np

class L2:
    def cost(self, x):
        return 0.5 * x * x

    def influence(self, x):
        return x

    def weight(self, x):
        return np.ones(x.size)

class L1:
    def cost(self, x):
        return np.abs(x)

    def influence(self, x):
        return np.sign(x)

    def weight(self, x):
        return 1. / np.abs(x)

class Cauchy:
    def __init__(self, k):
        self.k = k

    def cost(self, x):
        return (0.5*self.k**2) * np.log(1 + (x/self.k)**2)

    def influence(self, x):
        return x / (1 + (x/self.k)**2)

    def weight(self, x):
        return 1 / (1 + (x/self.k)**2)

class Huber:
    def __init__(self, k):
        self.k = k

    def cost(self, x):
        cost = np.zeros(x.size)
        leq_mask = np.abs(x) <= self.k
        ge_mask = np.abs(x) > self.k
        cost[leq_mask] = 0.5*x[leq_mask]**2
        cost[ge_mask] = self.k*(np.abs(x[ge_mask]) - 0.5*self.k)
        return cost

    def influence(self, x):
        infl = np.zeros(x.size)
        leq_mask = np.abs(x) <= self.k
        ge_mask = np.abs(x) > self.k
        infl[leq_mask] = x[leq_mask]
        infl[ge_mask] = self.k * np.sign(x[ge_mask])
        return infl

    def weight(self, x):
        wght = np.zeros(x.size)
        leq_mask = np.abs(x) <= self.k
        ge_mask = np.abs(x) > self.k
        wght[leq_mask] = np.ones(x[leq_mask].size)
        wght[ge_mask] = self.k / np.abs(x[ge_mask])
        return wght
