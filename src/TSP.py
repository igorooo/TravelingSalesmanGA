import math
import numpy as np
from random import shuffle, choice, randrange
import GA

#GA CONSTANTS
TOURNAMENT = 'TOURNAMENT'
ELITISM = 'ELITISM'
ROULETTE = 'ROULETTE'


# CONSTANTS USED IN FILE PARSING
COORD_SECTION = "NODE_COORD_SECTION"
DIMENSION = "DIMENSION:"

# euclidean distance p/q(nodeNumber, X, Y)
distance_func = lambda p, q : math.sqrt((q[1]-p[1])**2 + (q[2]-p[2])**2)

def parseCoordSection(lines):
    coordList = []
    for line in lines:
        temp = line.split()
        coordList.append((int(temp[0]), float(temp[1]), float(temp[2])))
    return coordList


def parseFile(filePath):
    file = open(filePath, "r")
    lines = file.readlines()
    lineIndex = 0

    for line in lines:
        strs = line.split()

        for str in strs:
            if str == DIMENSION:
                dimension = int(line.split()[1])
            if str == COORD_SECTION:
                return parseCoordSection(lines[lineIndex + 1:-2])
        lineIndex += 1


def countPopulationFitnes(pop):
    """

    :param pop: population
    :return: list of fitnes values for each individual in population
            fitnes value is equal to inverse of total distance (fit = 1 / distance)
    """
    popFitnes = []
    popSize = len(pop)
    for i in range(popSize):
        popFitnes.append(1/countTotalDistance(pop[i]))
    return popFitnes

def countTotalDistance(path):
    """
    count total distance between nodes in given path
    :param path: ordered list of nodes, Node - tuple(NodeID, CoordX, CoordY)
    :return: total distance
    """
    current = path[0]
    totalDistance = 0

    for node in path[1:]:
        totalDistance += distance_func(current, node)
        current = node

    return totalDistance

def TSPrandomAlgorithm(nodes, numberOfTrials):
    """
    Generating random solutions
    :param nodes:
    :param numberOfTrials:
    :return: (best found distance, path)
    """
    bestResult = countTotalDistance(nodes)
    bestPath = nodes[:]

    for i in range(numberOfTrials - 1):
        shuffle(nodes)
        totalDistance = countTotalDistance(nodes)
        if totalDistance < bestResult:
            bestResult = totalDistance
            bestPath = nodes[:]

    return (bestResult, bestPath)

def TSPforceAlgorithm(nodes, numberOfTrials):
    bestResult = float("+inf")
    bestPath = None

    for i in range(numberOfTrials):
        nodes_cp = nodes[:]
        startPoint = choice(nodes_cp)
        dist, path = getShortestPath(startPoint, nodes_cp)

        if dist < bestResult:
            bestResult = dist
            bestPath = path

    return bestResult, bestPath


def getShortestPath(startPoint, nodes):
    length = len(nodes)
    currentNode = startPoint
    path = []
    distance = 0

    for i in range(length):
        nodes.remove(currentNode)
        path.append(currentNode)
        if len(nodes) > 0:
            dist, currentNode = findNearestNode(currentNode, nodes)
            distance += dist
    return distance, path



def findNearestNode(node, nodes):
    bestDistance = float("+inf")
    nearestNode = None
    for nod in nodes:
        distance = distance_func(node, nod)
        if distance < bestDistance:
            bestDistance = distance
            nearestNode = nod
    return bestDistance, nearestNode


# ---------- GENETIC ALGORITHM METHODS ---------

def initializePopulation(popSize, nodes):
    pop = []
    for i in range(popSize):
        individual = nodes[:]
        shuffle(individual)
        pop.append(individual)
    return pop

def tournamentSelect(nSize, pop):
    pop = pop[:]
    newPop = []
    popSize = len(pop)
    popFitnes = countPopulationFitnes(pop)
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

def rouletteSelect(pop):
    pop = pop[:]
    newPop = []
    popSize = len(pop)
    popFitnes = countPopulationFitnes(pop)

    #generating probability distribution
    sum = np.sum(popFitnes)
    probDistr = [popFitnes[i] / sum for i in range(popSize)]
    inxArr = [i for i in range(popSize)]

    for i in range(popSize):
        # random choice based on given probability distribution
        shot = np.random.choice(a=inxArr, p=probDistr)
        newPop.append(pop[shot])
    return newPop


def eliteSelect(pop, m):
    """

    :param pop:
    :param m: <0; 100> percent
    :return:
    """
    pop = pop[:]
    newPop = []
    popSize = len(pop)
    popFitnes = countPopulationFitnes(pop)
    n = int(popSize * m / 100)
    for i in range(popSize):
        pop[i] = (pop[i], popFitnes[i])
    pop.sort(key=lambda tup: tup[1], reverse=True)
    for i in range(n):
        newPop.append(pop[i][0])
    return newPop


path = parseFile("../TSP/berlin11_modified.tsp")
#path = parseFile("../TSP/berlin52.tsp")

pop = GA.Population(10, path)



popp = pop.pop
newPop = pop.rouletteSelect()
nPop = pop.eliteSelect(20)

print('----------------')
for p in popp:
    print(p.path)
print('----------------')
for p in newPop:
    print(p.path)

print('----------------')
for p in nPop:
    print(p.path)




