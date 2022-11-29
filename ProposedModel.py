from Task import Task
from AgentModelNew import Robots
from ShippingPort import ShippingPort
import DynamicEnvironment as DE
import mesa
import random
import copy
import math
import numpy as np
from typing import List
from SelectiveRandomActivation import SelectiveRandomActivation

TaskList = List[Task]


class Model(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N: int, limitX: list, limitY: list, numTruck: int, numSrt: list, numMrt: list,
                 agentPayload: int, avgVelocity: float, numRobotsNeeded: int,
                 timeFactor: float, timeInfluFactor: float, encouragementFactor: float, numSRTInfluence: float,
                 mrtWeightBalanceFactor: float, waitThresholdFactor: float, mrtDistanceFactor: float, waitFactor: float,
                 acceptFactor: float,
                 numRobotsSpeculated: int, numKeepMRT: float,
                 numShipPort: int, weightRangeSrt: list, requiredRobots: int, randomSeed: int,
                 emergencyProportion: list):
        self.numAgents = N  # the number of robots
        self.limitX = limitX  # [min, max] of x
        self.limitY = limitY  # [min, max] of y
        self.numTruck = numTruck  # the number of trucks
        self.numSrt = numSrt  # number range of single-robot tasks
        self.numMrt = numMrt  # number range of multiple-robot tasks
        self.agentPayload = agentPayload  # loading capacity of agent
        self.numRobotsNeeded = numRobotsNeeded  # maximum number of robots needed
        self.avgVelocity = avgVelocity  # the average speed of robots

        self.timeFactor = timeFactor  # relevant with the importance of task's deadline
        self.timeInfluFactor = timeInfluFactor
        self.encouragementFactor = encouragementFactor
        self.numSRTInfluence = numSRTInfluence
        self.mrtWeightBalanceFactor = mrtWeightBalanceFactor
        self.mrtDistanceFactor = mrtDistanceFactor
        self.waitThresholdFactor = waitThresholdFactor
        self.waitFactor = waitFactor
        self.acceptFactor = acceptFactor

        self.numRobotsSpeculated = numRobotsSpeculated
        self.numKeepMRT = numKeepMRT

        self.numShipPort = numShipPort  # the number of Ship Port
        self.weightRangeSrt = weightRangeSrt  # the weight range of Srt

        self.shipPortPos = [[1, 50], [150, 1], [150, 99]]  # the shipping port position
        self.numTask = 0
        self.current_id = 0

        self.computationTime = []  # used to store computation time of task allocation for all robots
        self.computationTimeRobot = []  # used to store computation time of task allocation for robot 1
        self.recordTime = []  # store the recording time

        self.totalWaitTime = 0  # store the sum of wait time for all robots
        self.requiredRobots = requiredRobots
        self.emergencyProportion = emergencyProportion
        self.taskList, self.timeThreshold, self.numSrtList, self.numMrtList, self.truckScheduleList, self.numTask = DE.create_tasks(
            self,
            self.requiredRobots, self.agentPayload, self.numTruck, self.numSrt, self.numMrt, self.avgVelocity,
            self.limitX, self.limitY, self.numShipPort, self.numRobotsNeeded, self.weightRangeSrt,
            self.shipPortPos, randomSeed, emergencyProportion)

        self.numTaskList = [[self.numMrtList[i + self.numTruck * j] + self.numSrtList[i + self.numTruck * j] for i in
                             range(self.numTruck)]
                            for j in range(self.numShipPort)]

        self.sumSrt = sum(self.numSrtList)
        self.sumMrt = sum(self.numMrtList)

        self.truckDelayTime = 0
        self.truckDelayRate = 0
        self.srtDelayRate = 0
        self.mrtDelayRate = 0
        self.taskDelayRate = 0

        self.realTimeSchedule = copy.deepcopy(self.truckScheduleList)
        self.gridSize = 5
        self.length = int(self.limitX[1] / self.gridSize)
        self.width = int(self.limitY[1] / self.gridSize)
        self.grid = mesa.space.MultiGrid(self.length, self.width, False)
        self.schedule = mesa.time.BaseScheduler(self)
        # self.schedule = SelectiveRandomActivation(self)

        self.travelDistanceList = [0] * self.numAgents
        self.idleTimeList = [0] * self.numAgents

        self.running = True
        self.finish = False

        self.winningBidMatrix = []  # store the bidding value for all agents
        self.taskBundle = []  # store the selected tasks of all agents
        self.prevAvailableTimeMatrix = []
        self.currentAvailableTimeMatrix = []

        random.seed(randomSeed)

        # Create robots
        for i in range(self.numAgents):
            a = Robots(self.current_id, self)
            a.avgVelocity = avgVelocity
            self.current_id += 1
            self.schedule.add(a)
            # Initialise the position of robots
            cx = random.uniform(self.limitX[0], self.limitX[1])
            cy = random.uniform(self.limitY[0], self.limitY[1])
            dx, dy = self.convert_continuous_to_discrete([cx, cy])
            self.grid.place_agent(a, (dx, dy))
            self.schedule._agents[i].availablePos = (cx, cy)
            self.schedule._agents[i].continuousPos = (cx, cy)

        # Create shipping Port
        for i in range(self.numShipPort):
            a = ShippingPort(self.current_id, self, self.shipPortPos[i], self.numTruck, 0, 15, i)
            self.current_id += 1
            self.schedule.add(a)
            dx, dy = self.convert_continuous_to_discrete(self.shipPortPos[i])
            self.grid.place_agent(a, (dx, dy))

        # Create tasks
        for i in range(self.numShipPort):
            for task in self.taskList[i]:
                task.unique_id = self.current_id
                self.current_id += 1
                dx, dy = self.convert_continuous_to_discrete(task.pickPoint)
                task.dPos = (dx, dy)
                self.schedule.add(task)
                self.grid.place_agent(task, (dx, dy))

        self.datacollector = mesa.DataCollector(
            {
                "loading Point 1": lambda m: m.schedule._agents[self.numAgents].calculate_remaining_task(),
                "loading Point 2": lambda m: m.schedule._agents[self.numAgents + 1].calculate_remaining_task(),
                "loading Point 3": lambda m: m.schedule._agents[self.numAgents + 2].calculate_remaining_task(),
            }
        )
        self.datacollector.collect(self)
        self.bestAvailablePosList = []
        self.bestAvailableTimeList = []

    def convert_continuous_to_discrete(self, continuousPos: list):
        x = math.floor(continuousPos[0] / self.gridSize)
        y = math.floor(continuousPos[1] / self.gridSize)
        return x, y

    def find_index(self, task):
        for i in range(self.numAgents + self.numShipPort, self.current_id):
            if task == self.schedule._agents[i]:
                return i

    def bundle_remove(self, availableTask: TaskList, maxTaskDepth: int, idxAgent: int):
        """
        Update bundles after communication
        For outbid agents, releases tasks from bundles
        """
        # first round, eliminate the outbid tasks and tag the tasks in the task bundle whose available time need to be changed
        # available time is the task finishing time
        out_bid_for_task = False
        availableTimeChangeTag = []
        for idx in range(maxTaskDepth):
            # If bundle(j) < 0, it means that all tasks up to task j are
            # still valid and in paths, the rest (j to MAX_DEPTH) are released
            if self.taskBundle[idxAgent][idx] is None:
                break
            else:
                # Test if agent has been outbid for a task.  If it has, release it and all subsequent tasks in its path.
                taskIndex = availableTask.index(self.taskBundle[idxAgent][idx])
                if self.winningBidMatrix[idxAgent][taskIndex][idxAgent] == -1:
                    out_bid_for_task = True
                if out_bid_for_task:
                    # The agent has lost a previous task, release this one too
                    if self.winningBidMatrix[idxAgent][taskIndex][idxAgent] > 0:
                        self.winningBidMatrix[idxAgent][taskIndex][idxAgent] = -1
                    self.taskBundle[idxAgent][idx] = None
                else:
                    # compare the previous available time matrix and current available time matrix
                    if max(self.prevAvailableTimeMatrix[idxAgent][taskIndex]) != max(
                            self.currentAvailableTimeMatrix[idxAgent][taskIndex]):
                        availableTimeChangeTag.append(idx)
        if availableTimeChangeTag:
            for idx in range(availableTimeChangeTag[0], maxTaskDepth):
                if self.taskBundle[idxAgent][idx] is None:
                    break
                else:
                    currentTask = self.taskBundle[idxAgent][idx]
                    taskIndex = availableTask.index(currentTask)
                    if idx - 1 >= 0:
                        prevTask = self.taskBundle[idxAgent][idx - 1]
                        prevTaskIndex = availableTask.index(prevTask)
                        prevAvailableTime = self.currentAvailableTimeMatrix[idxAgent][prevTaskIndex][idxAgent]
                        prevPos = prevTask.deliveryPoint
                    else:
                        prevPos = self.bestAvailablePosList[idxAgent]
                    if idx != availableTimeChangeTag[0]:
                        modifiedTime = \
                            prevAvailableTime + math.dist(prevPos, currentTask.pickPoint) + math.dist(
                                currentTask.pickPoint, currentTask.deliveryPoint)
                        self.currentAvailableTimeMatrix[idxAgent][taskIndex][idxAgent] = max(modifiedTime, max(
                            self.currentAvailableTimeMatrix[idxAgent][taskIndex]))
                    else:
                        self.currentAvailableTimeMatrix[idxAgent][taskIndex][idxAgent] = max(
                            self.currentAvailableTimeMatrix[idxAgent][taskIndex])
                    self.winningBidMatrix[idxAgent][taskIndex][idxAgent] = \
                        self.score(currentTask, self.currentAvailableTimeMatrix[idxAgent][taskIndex][idxAgent], prevPos)

    def communicate(self, timeMat: list, iterIdx: int, availableMrt: TaskList, numSrt: int, numMrt: int):
        """
        Runs consensus between neighbors. Checks for conflicts and resolves among agents.
        """
        # timeMat is the matrix of time of updates from the current winners
        # iterIdx is the current iteration
        # winner bid matrix stores the senders' and self winner bid matrix
        # available tasks = available srt + available mrt
        graph = [[1] * self.numAgents for _ in range(self.numAgents)]
        for i in range(self.numAgents):
            graph[i][i] = 0

        time_mat_new = copy.deepcopy(timeMat)
        old_z = [[-1] * numSrt for _ in range(self.numAgents)]
        old_y = [[-1] * numSrt for _ in range(self.numAgents)]
        oldAvailableTimeList = [[-1] * numSrt for _ in range(self.numAgents)]
        mrtWinnerBidMatrix = []
        mrtAvailTimeMatrix = []
        for i in range(self.numAgents):
            # extract winner list and winner bid list
            matrix = self.winningBidMatrix[i]
            for j in range(numSrt):
                if max(matrix[j]) > 0:
                    old_z[i][j] = np.argmax(matrix[j])
                    old_y[i][j] = max(matrix[j])
                    if self.currentAvailableTimeMatrix[i][j][old_z[i][j]] != -1:
                        oldAvailableTimeList[i][j] = self.currentAvailableTimeMatrix[i][j][old_z[i][j]]
                    else:
                        print("error")

            mrtWinnerBidMatrix.append(matrix[numSrt:])
            mrtAvailTimeMatrix.append(self.currentAvailableTimeMatrix[i][numSrt:])

        oldMrtWinnerBidMatrix = copy.deepcopy(mrtWinnerBidMatrix)
        oldMrtAvailTimeMatrix = copy.deepcopy(mrtAvailTimeMatrix)

        # Copy data
        z = copy.deepcopy(old_z)
        y = copy.deepcopy(old_y)
        availableTimeList = copy.deepcopy(oldAvailableTimeList)

        epsilon = 10e-6

        # Start communication between agents
        # sender   = k
        # receiver = i
        # task     = j

        for k in range(self.numAgents):
            for i in range(self.numAgents):
                if graph[k][i] == 1:
                    for j in range(numSrt):
                        # Entries 1 to 4: Sender thinks he has the task
                        if old_z[k][j] == k:
                            # Entry 1: Update or Leave
                            if z[i][j] == i:
                                if (old_y[k][j] - y[i][j]) > epsilon:  # Update
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                    availableTimeList[i][j] = oldAvailableTimeList[k][j]
                                elif abs(old_y[k][j] - y[i][j]) <= epsilon:  # Equal scores
                                    if z[i][j] > old_z[k][j]:  # Tie-break based on smaller index
                                        z[i][j] = old_z[k][j]
                                        y[i][j] = old_y[k][j]
                                        availableTimeList[i][j] = oldAvailableTimeList[k][j]
                            # Entry 2: Update
                            elif z[i][j] == k:
                                z[i][j] = old_z[k][j]
                                y[i][j] = old_y[k][j]
                                availableTimeList[i][j] = oldAvailableTimeList[k][j]
                            # Entry 3: Update or Leave
                            elif z[i][j] > -1:
                                if timeMat[k][z[i][j]] > time_mat_new[i][z[i][j]]:  # Update
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                    availableTimeList[i][j] = oldAvailableTimeList[k][j]
                                elif (old_y[k][j] - y[i][j]) > epsilon:  # Update
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                    availableTimeList[i][j] = oldAvailableTimeList[k][j]
                                elif abs(old_y[k][j] - y[i][j]) <= epsilon:  # Equal scores
                                    if z[i][j] > old_z[k][j]:  # Tie-break based on smaller index
                                        z[i][j] = old_z[k][j]
                                        y[i][j] = old_y[k][j]
                                        availableTimeList[i][j] = oldAvailableTimeList[k][j]

                            # Entry 4: Update
                            elif z[i][j] == -1:
                                z[i][j] = old_z[k][j]
                                y[i][j] = old_y[k][j]
                                availableTimeList[i][j] = oldAvailableTimeList[k][j]
                            else:
                                print(z[i][j])
                                raise Exception("Unknown winner value: please revise!")

                        # Entries 5 to 8: Sender thinks receiver has the task
                        elif old_z[k][j] == i:

                            # Entry 5: Leave
                            if z[i][j] == i:
                                # Do nothing
                                pass

                            # Entry 6: Reset
                            elif z[i][j] == k:
                                z[i][j] = -1
                                y[i][j] = -1
                                availableTimeList[i][j] = -1
                            # Entry 7: Reset or Leave
                            elif z[i][j] > -1:
                                if timeMat[k][z[i][j]] > time_mat_new[i][
                                    z[i][j]]:  # Reset
                                    z[i][j] = -1
                                    y[i][j] = -1
                                    availableTimeList[i][j] = -1

                            # Entry 8: Leave
                            elif z[i][j] == -1:
                                # Do nothing
                                pass

                            else:
                                print(z[i][j])
                                raise Exception("Unknown winner value: please revise!")

                        # Entries 9 to 13: Sender thinks someone else has the task
                        elif old_z[k][j] > -1:
                            # Entry 9: Update or Leave
                            if z[i][j] == i:
                                if timeMat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]:
                                    if (old_y[k][j] - y[i][j]) > epsilon:
                                        z[i][j] = old_z[k][j]  # Update
                                        y[i][j] = old_y[k][j]
                                        availableTimeList[i][j] = oldAvailableTimeList[k][j]
                                    elif abs(old_y[k][j] - y[i][j]) <= epsilon:  # Equal scores
                                        if z[i][j] > old_z[k][j]:  # Tie-break based on smaller index
                                            z[i][j] = old_z[k][j]
                                            y[i][j] = old_y[k][j]
                                            availableTimeList[i][j] = oldAvailableTimeList[k][j]

                            # Entry 10: Update or Reset
                            elif z[i][j] == k:
                                if timeMat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]:  # Update
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                    availableTimeList[i][j] = oldAvailableTimeList[k][j]
                                else:  # Reset
                                    z[i][j] = -1
                                    y[i][j] = -1
                                    availableTimeList[i][j] = -1

                            # Entry 11: Update or Leave
                            elif z[i][j] == old_z[k][j]:
                                if timeMat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]:  # Update
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                    availableTimeList[i][j] = oldAvailableTimeList[k][j]

                            # Entry 12: Update, Reset or Leave
                            elif z[i][j] > -1:
                                if timeMat[k][z[i][j]] > time_mat_new[i][z[i][j]]:
                                    if timeMat[k][old_z[k][j]] >= time_mat_new[i][old_z[k][j]]:  # Update
                                        z[i][j] = old_z[k][j]
                                        y[i][j] = old_y[k][j]
                                        availableTimeList[i][j] = oldAvailableTimeList[k][j]
                                    elif timeMat[k][old_z[k][j]] < time_mat_new[i][old_z[k][j]]:  # Reset
                                        z[i][j] = -1
                                        y[i][j] = -1
                                        availableTimeList[i][j] = -1
                                    else:
                                        raise Exception("Unknown condition for Entry 12: please revise!")
                                else:
                                    if timeMat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]:
                                        if (old_y[k][j] - y[i][j]) > epsilon:  # Update
                                            z[i][j] = old_z[k][j]
                                            y[i][j] = old_y[k][j]
                                            availableTimeList[i][j] = oldAvailableTimeList[k][j]
                                        elif abs(old_y[k][j] - y[i][j]) <= epsilon:  # Equal scores
                                            if z[i][j] > old_z[k][j]:  # Tie-break based on smaller index
                                                z[i][j] = old_z[k][j]
                                                y[i][j] = old_y[k][j]
                                                availableTimeList[i][j] = oldAvailableTimeList[k][j]

                            # Entry 13: Update or Leave
                            elif z[i][j] == -1:
                                if timeMat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]:  # Update
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                    availableTimeList[i][j] = oldAvailableTimeList[k][j]

                            else:
                                raise Exception("Unknown winner value: please revise!")

                        # Entries 14 to 17: Sender thinks no one has the task
                        elif old_z[k][j] == -1:

                            # Entry 14: Leave
                            if z[i][j] == i:
                                # Do nothing
                                pass

                            # Entry 15: Update
                            elif z[i][j] == k:
                                z[i][j] = old_z[k][j]
                                y[i][j] = old_y[k][j]
                                availableTimeList[i][j] = oldAvailableTimeList[k][j]

                            # Entry 16: Update or Leave
                            elif z[i][j] > -1:
                                if timeMat[k][z[i][j]] > time_mat_new[i][z[i][j]]:  # Update
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                    availableTimeList[i][j] = oldAvailableTimeList[k][j]

                            # Entry 17: Leave
                            elif z[i][j] == -1:
                                # Do nothing
                                pass
                            else:
                                raise Exception("Unknown winner value: please revise!")

                            # End of table
                        else:
                            raise Exception("Unknown winner value: please revise!")

                    for j in range(numMrt):
                        for n in range(self.numAgents):
                            if mrtWinnerBidMatrix[i][j][n] > 0:
                                if k == n:
                                    mrtWinnerBidMatrix[i][j][n] = oldMrtWinnerBidMatrix[k][j][n]
                                    mrtAvailTimeMatrix[i][j][n] = oldMrtAvailTimeMatrix[k][j][n]

                    for j in range(numMrt):

                        for n in range(self.numAgents):
                            if oldMrtWinnerBidMatrix[k][j][n] > 0:
                                if i != n and (timeMat[k][n] >= time_mat_new[i][n] or n == k):
                                    if len(np.where(np.array(mrtWinnerBidMatrix[i][j]) != -1)[0]) < availableMrt[
                                        j].numRobotsNeeded:
                                        mrtWinnerBidMatrix[i][j][n] = oldMrtWinnerBidMatrix[k][j][n]
                                        mrtAvailTimeMatrix[i][j][n] = oldMrtAvailTimeMatrix[k][j][n]
                                    elif len(np.where(np.array(mrtWinnerBidMatrix[i][j]) != -1)[0]) == \
                                            availableMrt[j].numRobotsNeeded:
                                        allocatedIndex = np.where(np.array(mrtWinnerBidMatrix[i][j]) != -1)[0].tolist()
                                        allocatedValue = [mrtWinnerBidMatrix[i][j][idx] for idx in allocatedIndex]
                                        minimum = min(allocatedValue)
                                        minimumIndex = allocatedIndex[np.argmin(allocatedValue)]
                                        if minimum < oldMrtWinnerBidMatrix[k][j][n]:
                                            mrtWinnerBidMatrix[i][j][minimumIndex] = -1
                                            mrtAvailTimeMatrix[i][j][minimumIndex] = -1
                                            mrtWinnerBidMatrix[i][j][n] = oldMrtWinnerBidMatrix[k][j][n]
                                            mrtAvailTimeMatrix[i][j][n] = oldMrtAvailTimeMatrix[k][j][n]
                                    else:
                                        print("error")
                    # Update timestamps for all agents based on latest comm
                    for n in range(self.numAgents):
                        if (n != i) and (time_mat_new[i][n] < timeMat[k][n]):
                            time_mat_new[i][n] = timeMat[k][n]
                    time_mat_new[i][k] = iterIdx

        # After communication, update the winner bid matrix of all the robots
        # return a modified winner bid matrix
        for i in range(self.numAgents):
            for j in range(numSrt):
                self.winningBidMatrix[i][j] = [-1] * self.numAgents
                self.currentAvailableTimeMatrix[i][j] = [-1] * self.numAgents
                if y[i][j] > 0:
                    self.winningBidMatrix[i][j][z[i][j]] = y[i][j]
                    self.currentAvailableTimeMatrix[i][j][z[i][j]] = availableTimeList[i][j]
            self.winningBidMatrix[i][numSrt:] = copy.deepcopy(mrtWinnerBidMatrix[i])
            self.currentAvailableTimeMatrix[i][numSrt:] = copy.deepcopy(mrtAvailTimeMatrix[i])
        return time_mat_new

    def bundle_construction(self, availableSrt: list, availableMrt: list, idxAgent: int):
        new_bid_flag = False
        agent = self.schedule._agents[idxAgent]
        # Check if bundle is full, the bundle is full when bundle_full_flag is True
        index_array = np.where(np.array(self.taskBundle[idxAgent]) == None)[0]
        if len(index_array) > 0:
            bundle_full_flag = False
        else:
            bundle_full_flag = True

        availableTasks = availableSrt.copy() + availableMrt.copy()
        while not bundle_full_flag:
            # whether the task is executable
            logic = [False] * len(availableTasks)
            # if the MRT is full or SRT comes, the robot in the coalition with smallest profit will be replaced
            replacedRobotIndex = [-1] * len(availableTasks)
            length = len(np.where(np.array(self.taskBundle[idxAgent]) != None)[0])
            if length > 0:
                prevTask = self.taskBundle[idxAgent][length - 1]
                pos = prevTask.deliveryPoint
                taskIndex = availableTasks.index(prevTask)
                availableTime = self.currentAvailableTimeMatrix[idxAgent][taskIndex][idxAgent]
            else:
                # if there is no task in the task bundle, then the available position is used as the initial position
                pos = self.bestAvailablePosList[idxAgent]
                availableTime = self.bestAvailableTimeList[idxAgent]
            scoreList = []
            mrtPrioritisedScore = []
            m = 0
            for task in availableTasks:
                if not self.taskBundle[idxAgent].count(task):
                    scoreList.append(self.score(task, availableTime, pos))
                    if task.taskType:
                        numParticipants = len(np.where(np.array(self.winningBidMatrix[idxAgent][m]) != -1)[0])
                        mrtPrioritisedScore.append(self.score(task, availableTime, pos) * (1 + 0.05 * numParticipants))
                else:
                    scoreList.append(-1)
                    if task.taskType:
                        mrtPrioritisedScore.append(-1)
                m += 1

            for j in range(len(availableTasks)):
                task = availableTasks[j]
                winnerBid = []
                winnerIndex = []
                if scoreList[j] != 0:
                    if task.taskType:
                        for k in range(self.numAgents):
                            if self.winningBidMatrix[idxAgent][j][k] > 0:
                                winnerBid.append(self.winningBidMatrix[idxAgent][j][k])
                                winnerIndex.append(k)
                        if task.numRobotsNeeded > len(winnerBid):
                            # there are valid positions
                            logic[j] = True
                        elif task.numRobotsNeeded == len(winnerBid):
                            if scoreList[j] > min(winnerBid):
                                logic[j] = True
                                replacedRobotIndex[j] = winnerIndex[np.argmin(winnerBid)]
                            else:
                                logic[j] = False
                        else:
                            print("error")
                    else:
                        if scoreList[j] > max(self.winningBidMatrix[idxAgent][j]):
                            logic[j] = True
                            if max(self.winningBidMatrix[idxAgent][j]) == -1:
                                replacedRobotIndex[j] = -1
                            else:
                                replacedRobotIndex[j] = np.argmax(self.winningBidMatrix[idxAgent][j])
                        else:
                            logic[j] = False
                else:
                    logic[j] = False

            mrtPrioritisedScore = np.array(logic) * np.array(scoreList[:len(availableSrt)] + mrtPrioritisedScore)
            actualScore = np.array(logic) * np.array(scoreList)
            if max(mrtPrioritisedScore) != 0:
                print(max(mrtPrioritisedScore))
                selectedTaskIndex = np.argmax(mrtPrioritisedScore)
                self.taskBundle[idxAgent][index_array[0]] = availableTasks[selectedTaskIndex]
                self.winningBidMatrix[idxAgent][selectedTaskIndex][idxAgent] = actualScore[selectedTaskIndex]
                if replacedRobotIndex[selectedTaskIndex] > -1:
                    self.winningBidMatrix[idxAgent][selectedTaskIndex][replacedRobotIndex[selectedTaskIndex]] = -1
                new_bid_flag = True
                selectedTask = availableTasks[selectedTaskIndex]
                self.currentAvailableTimeMatrix[idxAgent][selectedTaskIndex][idxAgent] = \
                    availableTime + math.dist(pos, selectedTask.pickPoint) + math.dist(selectedTask.deliveryPoint,
                                                                                       selectedTask.pickPoint)
                # Check if bundle is full
                index_array = np.where(np.array(self.taskBundle[idxAgent]) == None)[0]
                if len(index_array) > 0:
                    bundle_full_flag = False
                else:
                    bundle_full_flag = True
            else:
                break

        return new_bid_flag

    def score(self, task: Task, availableTime: int, availablePos: list):

        priority = 4 - task.truckID
        finishTime = availableTime + math.dist(availablePos, task.pickPoint) + math.dist(task.pickPoint,
                                                                                         task.deliveryPoint)
        gap = finishTime - task.startTime
        maxValue = (self.numSrt[1] + self.numMrt[1] * self.numRobotsNeeded) / self.numAgents
        distance = math.dist(availablePos, task.pickPoint)
        score = priority * (maxValue - gap / self.timeThreshold) - distance / self.timeThreshold
        return score

    def solve(self, availableSrt: TaskList, availableMrt: TaskList):
        """
        Main CBGA Function, if this function converge, every robot can get a bundle of tasks
        """
        # Initialize working variables
        # Current iteration
        iterIdx = 1
        # Matrix of time of updates from the current winners
        timeMat = [[0] * self.numAgents for _ in range(self.numAgents)]
        iterPrev = 0
        done_flag = False

        maxTaskDepth = math.ceil((len(availableMrt) * 3 + len(availableSrt)) / self.numAgents) + 5
        self.winningBidMatrix = [[[-1] * self.numAgents for _ in range(len(availableSrt) + len(availableMrt))] for _ in
                                 range(self.numAgents)]
        self.currentAvailableTimeMatrix = copy.deepcopy(self.winningBidMatrix)
        self.prevAvailableTimeMatrix = copy.deepcopy(self.winningBidMatrix)
        self.taskBundle = [[None] * maxTaskDepth for _ in range(self.numAgents)]

        # Main CBBA loop (runs until convergence)
        while not done_flag:

            # 1. Communicate
            # Perform consensus on winning agents and bid values (synchronous)
            timeMat = self.communicate(timeMat, iterIdx, availableMrt, len(availableSrt), len(availableMrt))
            availableTask = availableSrt + availableMrt
            # 2. Run CBBA bundle building/updating
            # Run CBBA on each agent (decentralized but synchronous)
            for idxAgent in range(self.numAgents):
                # Update bundles after messaging to drop tasks that are outbid
                self.bundle_remove(availableTask, maxTaskDepth, idxAgent)
                # Bid on new tasks and add them to the bundle
                new_bid_flag = self.bundle_construction(availableSrt, availableMrt, idxAgent)
                # Update last time things changed
                # needed for convergence but will be removed in the final implementation
                if new_bid_flag:
                    iterPrev = iterIdx
            self.prevAvailableTimeMatrix = copy.deepcopy(self.currentAvailableTimeMatrix)
            # 3. Convergence Check
            # Determine if the assignment is over (implemented for now, but later this loop will just run forever)
            if (iterIdx - iterPrev) > self.numAgents:
                done_flag = True
            elif (iterIdx - iterPrev) > (2 * self.numAgents):
                print("Algorithm did not converge due to communication trouble")
                done_flag = True
            else:
                # Maintain loop
                iterIdx += 1

    def check_available_tasks(self):
        availableSrt = []
        availableMrt = []
        currentTime = self.schedule.time
        numDepot = self.numShipPort
        for i in range(numDepot):
            if self.taskList[i]:
                # check the first id of the task list
                truckName = self.taskList[i][0].truckID
                for task in self.taskList[i]:
                    if task.truckID == truckName:
                        if currentTime >= task.startTime:
                            # update tasks' state
                            index = self.find_index(task)
                            self.schedule._agents[index].appear = True
                            if not task.selected:
                                # include these tasks in available Srt list and Mrt list
                                if task.taskType == 0:
                                    availableSrt.append(task)
                                else:
                                    availableMrt.append(task)
                        else:
                            break
                    else:
                        break

        return availableSrt, availableMrt

    def tag_selected_tasks(self, availableTasks):
        for task in availableTasks:
            task.selected = True

    def step(self):

        # self.datacollector.collect(self)
        # print(self.schedule.time)
        # whether the task allocation process is finished
        taskAllocationToken = False
        if self.schedule.time == 0:
            self.bestAvailableTimeList = [0] * self.numAgents
            self.bestAvailablePosList = [self.schedule._agents[i].availablePos for i in range(self.numAgents)]

        availableSrt, availableMrt = self.check_available_tasks()
        availableTask = availableSrt + availableMrt
        # the number of idle robots exceeds some threshold and available task is not empty
        # if the number of available tasks is more than 10
        num = 0
        for availableTime in self.bestAvailableTimeList:
            if availableTime <= self.schedule.time:
                num += 1
            if num >= 1 / 2 * self.numAgents:
                taskAllocationToken = True
        if len(availableTask) > 10:
            taskAllocationToken = True

        self.bestAvailableTimeList = [max(self.bestAvailableTimeList[i], self.schedule.time) for i in
                                      range(self.numAgents)]

        if taskAllocationToken and availableTask:
            self.solve(availableSrt, availableMrt)
            # leave selected tag on the available tasks
            self.tag_selected_tasks(availableTask)

            bestCoalition = {}
            for j in range(len(availableMrt)):
                bestCoalition[availableMrt[j]] = np.where(np.array(self.winningBidMatrix[0][j]) != -1)[0].tolist()

            # update value
            bestTaskList = [[] for _ in range(self.numAgents)]
            for i in range(self.numAgents):
                # update agents' next task list
                self.schedule._agents[i].nextTask.append(self.taskBundle[i])
                # store the best coalition dictionary for every MRT
                self.bestCoalition.update(bestCoalition)
                indexList = np.where(np.array(availableTask) is not None)[0].tolist()
                index = availableTask.index(self.taskBundle[i][indexList[-1]])
                self.bestAvailableTimeList[i] = self.currentAvailableTimeMatrix[i][index][i]
                self.bestAvailablePosList[i] = self.taskBundle[i][index].deliveryPoint

        if not self.taskList[0] and not self.taskList[1] and not self.taskList[2]:
            self.finish = True

        # iterate for every robot
        if not self.finish:
            self.schedule.step()
            # deal with mrt movements, robots need to wait for other partners at the pickup point
            for i in range(self.numAgents):
                # if robot i is waiting at the pickup point
                agent = self.schedule._agents[i]
                if agent.wait:
                    j = 0
                    for partner in agent.currentPartners:
                        agent1 = self.schedule._agents[partner]
                        if agent1.wait and agent.ongoingTask[0] == agent1.ongoingTask[0]:
                            j += 1

                    if j == len(agent.currentPartners):
                        # all the partners are waiting at the pickup point
                        if agent.ongoingTask[0].taskType == 1:
                            # update the state of the task
                            index = self.find_index(agent.ongoingTask[0])
                            task = self.schedule._agents[index]
                            task.picked = True
                            task.appear = False
                        agent.pickOrDeliver = True
                        agent.workState = 3
                        agent.wait = False
                        agent.pathTag = 0
                        agent.targetPosMid = agent.deliveryPath[agent.pathTag]
                        agent.movement_calculation(agent.targetPosMid)
                        for partner in agent.currentPartners:
                            agent1 = self.schedule._agents[partner]
                            agent1.workState = 3
                            agent1.wait = False
                            agent1.pickOrDeliver = True
                            agent1.pathTag = 0
                            agent1.targetPosMid = agent1.deliveryPath[agent1.pathTag]
                            agent1.movement_calculation(agent1.targetPosMid)
