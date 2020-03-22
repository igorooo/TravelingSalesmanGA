import math
import numpy as np
from random import shuffle, choice, randrange

# euclidean distance p/q(nodeNumber, X, Y)
distance_func = lambda p, q: math.sqrt((q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2)

class Individual:

    def __init__(self, path):
        self.path = path
        self.countFitnes()

    def countFitnes(self):
        current = self.path[0]
        totalDistance = 0

        for node in self.path[1:]:
            totalDistance += distance_func(current, node)
            current = node

        self.fitnes = 1 / totalDistance
        self.distance = totalDistance


class Population:

    def __init__(self, popSize, nodes):
        # TODO better init
        self.pop = []
        for i in range(popSize):
            nodes = nodes[:]
            shuffle(nodes)
            individual = Individual(nodes)
            self.pop.append(individual)

    def tournamentSelect(self, nSize):
        pop = self.pop[:]
        newPop = []
        popSize = len(pop)
        popFitnes = [pop[i].fitnes for i in range(popSize)]
        for i in range(popSize):
            winner = 0
            currBest = 0
            for j in range(nSize):
                indiv = randrange(popSize)
                if popFitnes[indiv] >= currBest:
                    winner = indiv
                    currBest = popFitnes[indiv]
            newPop.append(pop[winner])
        return newPop

    def rouletteSelect(self):
        pop = self.pop[:]
        newPop = []
        popSize = len(pop)
        popFitnes = [pop[i].fitnes for i in range(popSize)]

        # generating probability distribution
        sum = np.sum(popFitnes)
        probDistr = [popFitnes[i] / sum for i in range(popSize)]
        inxArr = [i for i in range(popSize)]

        for i in range(popSize):
            # random choice based on given probability distribution
            shot = np.random.choice(a=inxArr, p=probDistr)
            newPop.append(pop[shot])
        return newPop

    def eliteSelect(self, m):
        """
            :param pop:
            :param m: <0; 100> percent
            :return:
            """
        pop = self.pop[:]
        newPop = []
        popSize = len(pop)
        popFitnes = [pop[i].fitnes for i in range(popSize)]
        n = int(popSize * m / 100)
        for i in range(popSize):
            pop[i] = (pop[i], popFitnes[i])
        pop.sort(key=lambda tup: tup[1], reverse=True)
        for i in range(n):
            newPop.append(pop[i][0])
        return newPop



