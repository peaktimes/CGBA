import numpy
import math
from heapq import heappop, heappush
import random


def generate_dm(initialPos: list, pickupList: list, deliveryList: list):
    posList = []
    posList.append(initialPos)
    for pos in pickupList:
        posList.append(pos)
    for pos in deliveryList:
        posList.append(pos)
    dm = [[999999] * len(posList) for _ in range(len(posList))]
    for i in range(len(posList)):
        for j in range(len(posList)):
            if i == 0 and 0 < j < (1 + len(pickupList)):
                dm[i][j] = math.dist(posList[i], posList[j])
            if 0 < i < (1 + len(pickupList)) and 0 < j < len(posList):
                if i != j:
                    dm[i][j] = math.dist(posList[i], posList[j])
            if len(pickupList) < i < len(posList) and len(pickupList) < j < len(posList):
                if i != j:
                    dm[i][j] = math.dist(posList[i], posList[j])
    return dm, posList


def initialise_route(pickupList: list, deliveryList: list, posList: list):
    a = list(range(1, 1 + len(pickupList)))
    b = list(range(1 + len(pickupList), len(posList)))
    random.shuffle(a)
    random.shuffle(b)
    res = [0] + a + b
    return res


def evaluate(dm: list, solution: list):
    res = [dm[u][v] for u, v in zip(solution[:-1], solution[1:])]
    return sum(res)


def modify(current: list, lenPickup: int, lenDelivery: int):
    new = current.copy()
    index_a = random.randint(1, len(current) - 1)
    if lenPickup < index_a < len(current):
        index_b = random.randint(lenPickup + 1, len(current) - 1)
        if lenDelivery > 1:
            while index_b == index_a:
                index_b = random.randint(lenPickup + 1, len(current) - 1)
    else:
        index_b = random.randint(1, lenPickup)
        if lenPickup > 1:
            while index_b == index_a:
                index_b = random.randint(1, lenPickup)
    new[index_a], new[index_b] = new[index_b], new[index_a]
    return new


def sa_path(nodeCount: int, initialPos: list, pickupList: list, deliveryList: list):
    initialTemperature = (math.floor(nodeCount / 3) + 1) * 3
    temperatureDecay = 0.95
    stoppingTemperature = 1
    # generate dm
    dm, posList = generate_dm(initialPos, pickupList, deliveryList)
    currentSolution = initialise_route(pickupList, deliveryList, posList)
    currentScore = evaluate(dm, currentSolution)
    bestScore = currentScore
    bestSolution = currentSolution
    temperature = initialTemperature
    while temperature > stoppingTemperature:
        newSolution = modify(currentSolution, len(pickupList), len(deliveryList))
        newScore = evaluate(dm, newSolution)
        if newScore < bestScore:
            bestScore = newScore
            bestSolution = newSolution
        if newScore < currentScore:
            currentSolution = newSolution
            currentScore = newScore
        else:
            delta = (newScore - currentScore) / currentScore
            probability = math.exp(-delta / temperature) / 2.5
            if probability > random.uniform(0, 1):
                currentSolution = newSolution
                currentScore = newScore
        temperature *= temperatureDecay
    return bestScore, bestSolution