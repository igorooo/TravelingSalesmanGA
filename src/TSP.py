import math
import numpy as np
from random import shuffle, choice, randrange
import csv
import GA
import matplotlib.pyplot as plt

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


def writeToCsv(fileName, results):
    with open('../results/' + fileName + '.csv', 'w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)

def plotResults(title, results, rndResult= None, forceResult=None):
    bestD, avgD, worstD, std, rnd, frc = [], [], [], [], [], []
    for i in range(len(results)):
        bestD.append(results[i][0])
        avgD.append(results[i][1])
        worstD.append(results[i][2])
        std.append(results[i][3])

        if rndResult != None and forceResult != None:
            rnd.append(rndResult)
            frc.append(forceResult)

    T = [i for i in range(0, GA.GEN)]
    plt.figure().suptitle(title, fontsize=10)
    plt.plot(T, bestD, label='best distance')
    plt.plot(T, avgD, label='average distance')
    plt.plot(T, worstD, label='worst distance')
    plt.plot(T, std, label='std')

    if rndResult != None and forceResult != None:
        plt.plot(T, rnd, label='random algorithm best result')
        plt.plot(T, frc, label='force algorithm best result')

    plt.legend()
    plt.show()
    plt.clf()

def setGensNumber(level):
    if level == 'EASY':
        GA.GEN = 1000
    elif level == 'MID':
        GA.GEN = 5000
    elif level == 'MID+':
        GA.GEN = 7000
    else:
        GA.GEN = 10000



if __name__ == "__main__":
    nodes = {
        'berlin11_modified': (parseFile("../TSP/berlin11_modified.tsp"), 'EASY'),
        'berlin52' : (parseFile("../TSP/berlin52.tsp"), 'EASY'),
        'kroA100': (parseFile("../TSP/kroA100.tsp"), 'MID'),
        'kroA150': (parseFile("../TSP/kroA150.tsp"), 'MID'),
        'kroA200' : (parseFile("../TSP/kroA200.tsp"), 'MID'),
        'fl417': (parseFile("../TSP/fl417.tsp"), 'MID+'),
        'gr666': (parseFile("../TSP/gr666.tsp"), 'MID+'),
        'nrw1379': (parseFile("../TSP/nrw1379.tsp"), 'HARD'),
        'pr2392': (parseFile("../TSP/pr2392.tsp"), 'HARD')
    }

    for k, v in nodes.items():
        ga = GA.GA(v[0])
        setGensNumber(v[1])
        bestRandomAlgResult = TSPrandomAlgorithm(v[0], GA.GEN*GA.POP_SIZE)
        bestForceAlgResult = TSPforceAlgorithm(v[0], GA.GEN)
        stats = ga.startGA(crossMethod=GA.OX, selectMethod='tournament', mutMethod='INV')
        plotResults(k, stats, bestRandomAlgResult[0], bestForceAlgResult[0])
        writeToCsv(k, stats)
        writeToCsv(k+'classic', [[bestRandomAlgResult[0], bestForceAlgResult[0]]])


