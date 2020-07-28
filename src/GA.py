import math
import numpy as np
from random import shuffle, choice, randrange, uniform
from copy import copy, deepcopy

# GA constants
POP_SIZE = 100
GEN = 1000
Px = 0.75
Pm = 0.10
TOUR = 6

# euclidean distance p/q(nodeNumber, X, Y)
distance_func = lambda p, q: math.sqrt((q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2)

class Individual:

    def __init__(self, path):
        self.path = path
        self.length = len(path)
        self.countFitnes()

    def MUT_SWAP(self):
        j = randrange(1, self.length)
        i = randrange(j)
        node = self.path[i]
        self.path[i] = self.path[j]
        self.path[j] = node
        self.countFitnes()

    def MUT_INV(self):
        j = randrange(1, self.length)
        i = randrange(j)
        nodes = self.path[i:j]
        nodes.reverse()
        self.path[i:j] = nodes
        self.countFitnes()

    def countFitnes(self):
        current = self.path[0]
        totalDistance = 0

        for node in self.path[1:]:
            totalDistance += distance_func(current, node)
            current = node

        self.fitnes = 1 / totalDistance
        self.distance = totalDistance

    def __str__(self):
        string = ""
        for i in range(self.length):
            string += '|' + str(self.path[i][0])
        return string


class Population:

    def __init__(self, popSize=0, nodes=None, pop=None):
        # TODO better init
        if pop != None:
            self.pop = pop
            self.popSize = len(self.pop)
        else:
            self.pop = []
            self.popSize = popSize
            for i in range(popSize):
                nodes = nodes[:]
                shuffle(nodes)
                individual = Individual(nodes)
                self.pop.append(individual)
        self.probDistr = None


    def getPopulationStats(self):
        bestF = 0
        bestD = 0
        bestInd = None
        worstF = float('+inf')
        worstD = 0
        worstInd = None
        avarageDistance = 0

        popFitnes = [self.pop[i].fitnes for i in range(self.popSize)]
        popDistance = [self.pop[i].distance for i in range(self.popSize)]

        for i in range(len(self.pop)):
            avarageDistance += self.pop[i].distance
            if self.pop[i].fitnes >= bestF:
                bestInd = self.pop[i]
                bestF = self.pop[i].fitnes
                bestD = self.pop[i].distance
            if self.pop[i].fitnes < worstF:
                worstInd = self.pop[i]
                worstF = self.pop[i].fitnes
                worstD = self.pop[i].distance
        avarageDistance /= self.popSize
        return [bestD, avarageDistance, worstD, np.std(popDistance)]



    def tournamentSelect(self, nSize=3):
        winner = None
        currBest = 0
        for j in range(nSize):
            indiv = choice(self.pop)
            if indiv.fitnes >= currBest:
                winner = indiv
                currBest = indiv.fitnes
        return copy(winner)

    def rouletteSelectPrepare(self):

        popFitnes = [self.pop[i].fitnes for i in range(self.popSize)]

        # generating probability distribution
        sum = np.sum(popFitnes)
        self.probDistr = [popFitnes[i] / sum for i in range(self.popSize)]
        self.inxArr = [i for i in range(self.popSize)]


    def rouletteSelect(self):
        """
        Neet to call self.rouletteSelectPrepare to create probability distribution array before calling roulette select
        :return: individual
        """
        if self.probDistr == None:
            self.rouletteSelectPrepare()
        shot = np.random.choice(a=self.inxArr, p=self.probDistr)
        return copy(self.pop[shot])


    def eliteSelect(self, m):
        """
            :param m: <0; 100> percent
            :return: list with best individuals from population
            """
        pop = self.pop[:]
        newPop = []
        popFitnes = [pop[i].fitnes for i in range(self.popSize)]
        n = int(self.popSize* m / 100)
        for i in range(self.popSize):
            pop[i] = (pop[i], popFitnes[i])
        pop.sort(key=lambda tup: tup[1], reverse=True)
        for i in range(n):
            newPop.append(pop[i][0])
        return newPop

def OX(P1, P2):

    n = randrange(1, P1.length)
    m = randrange(n)
    nodes = P1.path[m:n]
    p2Order = P2.path[:]

    # remove elements from selected region
    for nod in nodes:
        for node in p2Order:
            if node[0] == nod[0]:
                p2Order.remove(node)

    res = P1.path[:]
    res[m:n] = nodes
    res[:m] = p2Order[:m]
    res[n:] = p2Order[m:]

    indv1 = Individual(res)

    tmp = P1
    P1 = P2
    P2 = tmp
    nodes = P1.path[m:n]
    p2Order = P2.path[:]
    # remove elements from selected region
    for nod in nodes:
        for node in p2Order:
            if node[0] == nod[0]:
                p2Order.remove(node)
    res = P1.path[:]
    res[m:n] = nodes
    res[:m] = p2Order[:m]
    res[n:] = p2Order[m:]

    indv2 = Individual(res)

    """ DEBUG
    for nod in nodes:
        print("|" + str(nod[0]), end='')
    print(';')
    for nod in p2Order:
        print("|" + str(nod[0]), end='')

    print("----OX-----")
    print("M: %d  N: %d" % (m,n))
    print("P1: " + str(P1))
    print("P2: " + str(P2))
    print("Res: " + str(indv))"""


    return indv1, indv2

def PMX(P1, P2):

    """ DEBUG
    testNod1 = [(2,1,0),(11,1,0),(5,0,0),(1,0,0),(8,0,0),(9,0,0),(10,0,0),(4,0,0),(7,0,0),(3,0,0),(6,0,0)]
    testNod2 = [(11, 1, 0), (3, 0, 0), (7, 0, 0), (1, 0, 0), (5, 0, 0), (4, 0, 0), (10, 0, 0), (2, 1, 0), (6, 0, 0),
                (8, 0, 0), (9, 0, 0)]
    P1 = Individual(testNod1)
    P2 = Individual(testNod2)"""

    n = randrange(1 ,P1.length)
    m = randrange(n)
    p1nodes = P1.path[m:n]
    p2nodes = P2.path[m:n]
    p1Mapper = {}
    p2Mapper = {}
    p1Child = P1.path[:]
    p2Child = P2.path[:]

    # initialize mapper
    for i in range(P1.length):
        p1Mapper[P1.path[i]] = None
        p2Mapper[P1.path[i]] = None


    for i in range(len(p1nodes)):
        p2Mapper[p1nodes[i]] = p2nodes[i]

    for key, value in p2Mapper.items():
        if value != None:
            p1Mapper[value] = key

    """ DEBUG
    print("--------PMX-----")
    print("M: %d, N: %d" % (m, n))
    print("P1: " + str(P1))
    print("P2: " + str(P2))
    print("p1Mapper: ")
    for key, value in p1Mapper.items():
        print(key, end='->')
        print(value)
    print("p2Mapper: ")
    for key, value in p2Mapper.items():
        print(key, end='->')
        print(value)"""

    for i in range(m):
        k = P1.path[i]
        while p1Mapper[k] != None and k != p1Mapper[k]:
            k = p1Mapper[k]
        p1Child[i] = k

        k = P2.path[i]
        while p2Mapper[k] != None and k != p1Mapper[k]:
            k = p2Mapper[k]
        p2Child[i] = k

    for i in range(n, P1.length):
        k = P1.path[i]
        while p1Mapper[k] != None and k != p1Mapper[k]:
            k = p1Mapper[k]
        p1Child[i] = k

        k = P2.path[i]
        while p2Mapper[k] != None and k != p1Mapper[k]:
            k = p2Mapper[k]
        p2Child[i] = k

    p1Child[m:n] = p2nodes
    p2Child[m:n] = p1nodes
    child1 = Individual(p1Child)
    child2 = Individual(p2Child)

    """ DEBUG
    print("Res: " + str(child1))
    print("Res: " + str(child2))"""

    return child1, child2

class GA:

    def __init__(self, nodes):
        self.stats = []
        self.generations = []
        self.generations.append(Population(popSize=POP_SIZE,nodes=nodes))
        self.stats.append(self.generations[0].getPopulationStats())

    def startGA(self, selectMethod='roulette', crossMethod=OX, mutMethod='INV'):
        stats = []

        for i in range(GEN):
            newPop = []
            pop = self.generations[i]
            for j in range(int(POP_SIZE/2)):
                P1, P2, C1, C2 = None, None, None, None
                if selectMethod == 'tournament':
                    P1 = pop.tournamentSelect(TOUR)
                    P2 = pop.tournamentSelect(TOUR)

                elif selectMethod == 'roulette':
                    P1 = pop.rouletteSelect()
                    P2 = pop.rouletteSelect()

                else:
                    raise Exception("Wrong selection method")

                prob = uniform(0, 1)
                if prob < Px:
                    C1, C2 = crossMethod(P1, P2)
                if prob >= Px:
                    C1, C2 = copy(P1), copy(P2)

                newPop.append(C1)
                newPop.append(C2)

            for j in range(POP_SIZE):

                if mutMethod == 'INV':
                    prob = uniform(0, 1)
                    if prob < Pm:
                        newPop[j].MUT_INV()

                else:
                    for i in range(newPop[j].length):
                        prob = uniform(0, 1)
                        if prob < (Pm / 10): # ~0.01
                            newPop[j].MUT_SWAP()
            newPopulation = Population(pop=newPop)
            stats.append(newPopulation.getPopulationStats())
            self.generations.append(newPopulation)

        return stats
