import math
import random
import numpy as np
from Task import Task
from typing import List
import mesa

TaskList = List[Task]


def duration_calculation(numRobotsNeededList: list, numSRT: int, numMRT: int, requiredRobots: int, timeThreshold: float):
    numRobotsNeeded = 0
    for i in range(numMRT):
        numRobotsNeeded += numRobotsNeededList[i]
    numRobotsNeeded += numSRT
    duration = math.ceil(numRobotsNeeded / requiredRobots * timeThreshold)
    return duration


def generate_srt(uniqueId: int, model: mesa.Model, endTime: int, limit_x: list, limit_y: list, start_time: int,
                 depot: list, truckID: int,
                 weightScale: list, taskID: int, depotName: int):
    task = Task(uniqueId, model)
    task.taskType = 0
    task.loadOrUnload = 0  # load is 0, unload is 1
    task.taskID = taskID
    task.truckID = truckID
    task.startTime = start_time
    task.endTime = endTime
    if task.loadOrUnload == 0:
        task.pickPoint[0] = random.randint((limit_x[0] + 1), limit_x[1] - 1)  # x position
        task.pickPoint[1] = random.randint(limit_y[0] + 1, limit_y[1] - 1)  # y position
        task.deliveryPoint = depot
    else:
        task.pickPoint = depot
        task.deliveryPoint[0] = random.randint((limit_x[0] + 1), limit_x[1] - 1)  # x position
        task.deliveryPoint[1] = random.randint(limit_y[0] + 1, limit_y[1] - 1)  # y position
    task.weight = random.randint(weightScale[0] + 1, weightScale[1])
    task.numRobotsNeeded = 1
    task.depotName = depotName
    return task


def generate_mrt(uniqueId: int, model: mesa.Model, endTime: int, limit_x: list, limit_y: list, start_time: int,
                 depot: list, truckID: int,
                 taskID: int, loadingCapacity: int, numRobotsNeeded: int, depotName: int):
    task = Task(uniqueId, model)
    task.loadOrUnload = 0  # load is 0, unload is 1
    task.taskType = 1
    task.taskID = taskID
    task.truckID = truckID
    task.taskPriority = 1
    task.startTime = start_time
    task.numRobotsNeeded = numRobotsNeeded
    task.endTime = endTime
    if task.loadOrUnload == 0:
        task.pickPoint[0] = random.randint((limit_x[0] + 1), limit_x[1] - 1)  # x position
        task.pickPoint[1] = random.randint(limit_y[0] + 1, limit_y[1] - 1)  # y position
        task.deliveryPoint = depot
    else:
        task.pickPoint = depot
        task.deliveryPoint[0] = random.randint((limit_x[0] + 1), limit_x[1] - 1)  # x position
        task.deliveryPoint[1] = random.randint(limit_y[0] + 1, limit_y[1] - 1)  # x position
    task.weight = random.randint((numRobotsNeeded - 1) * loadingCapacity, numRobotsNeeded * loadingCapacity)
    task.depotName = depotName
    return task


def create_tasks(model: mesa.Model, requiredRobots: int, agentPayload: int, numTruck: int, num_srt: list, num_mrt: list,
                 avgVelocity: float, limitX: list, limitY: list, numShipPort: int,
                 numRobotsNeeded: int, weightRangeSrt: list, shipPortPos: list, randomSeed: int, emergencyProportion: list):
    taskList0: TaskList = []
    taskList1: TaskList = []
    taskList2: TaskList = []

    truckSchedule0 = [[0] * 2 for _ in range(numTruck)]
    truckSchedule1 = [[0] * 2 for _ in range(numTruck)]
    truckSchedule2 = [[0] * 2 for _ in range(numTruck)]

    truckScheduleList = {
        0: truckSchedule0,
        1: truckSchedule1,
        2: truckSchedule2,
    }

    taskList = {
        0: taskList0,
        1: taskList1,
        2: taskList2
    }
    # the maximum time for a round trip
    timeThreshold = math.dist([0, 0], [limitX[1], limitY[1]]) * 2 / avgVelocity

    numSrtList = []
    numMrtList = []

    random.seed(randomSeed)

    # generate random timestamp for every depot
    uniqueId = 0

    for i in range(numShipPort):
        depotName = i
        # iterate for every depot
        timeDepot = 0
        for j in range(numTruck):
            # iterate for every truck

            # store the number of robots needed for every Mrt in this truck
            numRobotsNeededList = []
            # generate random number of SRT
            numSRT = random.randint(num_srt[0], num_srt[1])
            # generate random number of MRT
            numMRT = random.randint(num_mrt[0], num_mrt[1])
            numSrtList.append(numSRT)
            numMrtList.append(numMRT)

            # the number of emergency Srt and Mrt
            emergencyNumberSrt = [0, 0]
            emergencyNumberMrt = [0, 0]
            emergencyNumberSrt[0] = math.ceil(numSRT * emergencyProportion[0])
            emergencyNumberSrt[1] = math.ceil(numSRT * emergencyProportion[1])
            emergencyNumberMrt[0] = math.ceil(numMRT * emergencyProportion[0])
            emergencyNumberMrt[1] = math.ceil(numMRT * emergencyProportion[1])
            emergencySrt = random.randint(emergencyNumberSrt[0], emergencyNumberSrt[1])
            emergencyMrt = random.randint(emergencyNumberMrt[0], emergencyNumberMrt[1])

            # generate the number of robots needed for Mrt
            for k in range(numMRT):
                numRobotsNeededList.append(random.randint(2, numRobotsNeeded))
            # generate the duration for the truck
            durationTruck = duration_calculation(numRobotsNeededList, numSRT, numMRT, requiredRobots,
                                                 timeThreshold)
            truckScheduleList[i][j] = [timeDepot, (timeDepot + durationTruck)]
            # generate the single robot tasks on the truck
            for k in range(numSRT):
                startTime = timeDepot
                endTime = timeDepot + durationTruck
                if k >= numSRT - emergencySrt:
                    startTime = random.randint(0, durationTruck - int(timeThreshold) * 2) + timeDepot
                taskList[depotName].append(
                    generate_srt(uniqueId, model, endTime, limitX, limitY, startTime,
                                 shipPortPos[i], j, weightRangeSrt, k,
                                 depotName))
                uniqueId += 1
            # generate the multiple robot tasks on the truck
            for k in range(numMRT):
                startTime = timeDepot
                endTime = timeDepot + durationTruck
                if k >= numMRT - emergencyMrt:
                    startTime = random.randint(0, durationTruck - int(timeThreshold) * 2) + timeDepot
                taskList[depotName].append(
                    generate_mrt(uniqueId, model, endTime, limitX, limitY, startTime,
                                 shipPortPos[i], j, k, agentPayload,
                                 numRobotsNeededList[k], depotName))
                uniqueId += 1
            # update the time
            timeDepot += durationTruck
            # timeDepot += random.randint(worldInfoInput.idleTime[0], worldInfoInput.idleTime[1])
    print(numSrtList)
    print(numMrtList)
    print(taskList)
    print(uniqueId)
    return taskList, timeThreshold, numSrtList, numMrtList, truckScheduleList, uniqueId