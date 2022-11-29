import mesa
import math
import random
from Task import Task
import numpy as np
import networkx as nx
from typing import List
from networkx.algorithms.bipartite.matrix import biadjacency_matrix
import SimulatedAnnealing as SA
import itertools
import time as MyTime

TaskList = List[Task]


class Robots(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.avgVelocity: float = 1  # agent cruise velocity (m/s)
        self.payload: int = 50  # the loading capacity of robots (kg)
        self.state: bool = True  # busy:1, idle:0

        self.availableTime: int = 0  # time when robots transfer from busy to idle
        self.availablePos: list = [0, 0]  # the robot's position at the available time

        self.currentPartners: list = []  # store the unique id of partners for ongoing tasks
        self.futurePartners: list = []  # store the unique id of partners for the next task

        self.ongoingTask: TaskList = []  # the ongoing task list for robot, it can be a list of Srt or a Mrt
        self.nextTask: TaskList = []  # the next allocated task for Robots

        self.pickupPath: list = []  # store the order of pickup point
        self.deliveryPath: list = []  # store the order of delivery point
        self.pathTag: int = 0  # path location tag

        self.angle: float = 0  # the movement angle and distance
        self.deltaX: float = 0
        self.deltaY: float = 0

        self.workState: int = 0  # no objects on it, single objects on it, multiple objects on it, MRT

        self.continuousPos: list = [0, 0]  # the continuous position of agent

        # whether the robot is waiting at the pickup point, the robot's pickOrDeliver
        # symbol will only become true when all its partners are in the waiting mode
        self.wait: bool = False
        self.targetPosMid: list = [0, 0]  # the target position in the current time
        self.pickOrDeliver: bool = False  # False: the robot is on its way to pick an item

        self.remainingCapacity = 50  # the remaining capacity of robots
        self.runTime = 0        #
    # other robots can modify my available time, position, future plan,
    # but I need to decide the ongoing task and position
    # In every time tick, I need to check whether my ongoing task is empty
    def step(self):

        time = self.model.schedule.time
        agents = self.model.schedule._agents
        uniqueId = self.unique_id
        ongoingTaskList = [agents[i].ongoingTask for i in range(self.model.numAgents)]
        nextTaskList = [agents[i].nextTask for i in range(self.model.numAgents)]

        if not self.ongoingTask:
            # when robot finishes its ongoing task, it needs to update its state
            self.pickOrDeliver = False
            self.wait = False
            self.state = False
            self.pathTag = 0
            self.pickupPath = []
            self.deliveryPath = []
            self.workState = 0

            if self.nextTask:
                self.ongoingTask.append(self.nextTask[0])
                self.currentPartners = self.futurePartners[0]
                self.nextTask.pop(0)
                self.futurePartners.pop(0)
                self.state = True
                # 2d matrix
                self.pickupPath.append(self.ongoingTask[0].pickPoint)
                self.deliveryPath.append(self.ongoingTask[0].deliveryPoint)
                self.targetPosMid = self.pickupPath[self.pathTag]
                self.movement_calculation(self.targetPosMid)

            # if next task is empty
            if not self.ongoingTask:
                newTask = False
                startTime: float = 0
                startTime = MyTime.time()
                self.availableTime = max(self.availableTime, time)
                # check available tasks
                availableSrt, availableMrt, newTask = self.check_available_tasks()
                # task allocation, update the robot's state
                if newTask or self.runTime <= time:
                    self.task_allocation(availableSrt, availableMrt)

                # update the robot's available time
                self.availableTime = max(self.availableTime, time)
                # if robot is allocated with a task
                if self.ongoingTask:
                    print("system time is " + str(self.model.schedule.time))
                    print("my available time is " + str(self.availableTime))
                    print("robot " + str(self.unique_id) + " pick " + str(self.ongoingTask))
                    if self.ongoingTask[0].taskType == 1:
                        # if ongoing task is multiple-robot task
                        print(self.currentPartners)

                    # delete the selected tasks from task list
                    for task in self.ongoingTask:
                        depotName = task.depotName
                        truckName = task.truckID
                        index = self.model.taskList[depotName].index(task)
                        self.model.taskList[depotName][index].selected = True
                        self.model.numTaskList[depotName][truckName] -= 1
                        # if task.endTime < self.model.schedule.time:
                        #     print("error")
                        # update the real-time schedule
                        self.modify_task_schedule(task, depotName, truckName)

                self.model.computationTime.append(MyTime.time() - startTime)
                if self.unique_id == 0:
                    self.model.computationTimeRobot.append(MyTime.time() - startTime)
                    self.model.recordTime.append(self.model.schedule.time)
        # if ongoing task is not empty:
        if self.ongoingTask:
            self.mov()

    def task_to_index(self, taskList: TaskList, targetPos: list, pickOrDeliver: bool):
        # find the corresponding task according to the targetPosMid
        index = []
        taskDoneList = []
        for task in taskList:
            if pickOrDeliver == 0:
                if task.pickPoint == targetPos:
                    # find the corresponding index in the schedule
                    index.append(self.find_index(task))
            else:
                if task.deliveryPoint == targetPos:
                    # find the corresponding index in the schedule
                    index.append(self.find_index(task))
                    taskDoneList.append(task)
        return index, taskDoneList

    def mov(self):
        # if robot is on its way to pick up items
        if not self.pickOrDeliver:
            # if robots can reach target pos in this tick
            if math.dist(self.continuousPos, self.targetPosMid) < self.avgVelocity:
                # update the continuous position of robots and move robots in the discrete grid map
                self.continuousPos = list(self.targetPosMid)
                dx, dy = self.model.convert_continuous_to_discrete(self.targetPosMid)
                self.model.grid.move_agent(self, (dx, dy))

                # update the next target position
                if self.ongoingTask[0].taskType == 0:

                    index, taskDoneList = self.task_to_index(self.ongoingTask, self.targetPosMid, self.pickOrDeliver)
                    # update the task state
                    for i in range(len(index)):
                        self.model.schedule._agents[index[i]].picked = True

                    # if robots have already picked all the items
                    if self.pathTag == len(self.pickupPath) - 1:
                        if self.pathTag == 0:
                            self.workState = 1
                        else:
                            self.workState = 2
                        self.pathTag = 0
                        self.pickOrDeliver = True
                        # update the target position after reach the former target position
                        self.targetPosMid = self.deliveryPath[self.pathTag]
                        self.movement_calculation(self.targetPosMid)
                    else:
                        if self.pathTag == 0:
                            self.workState = 1
                        else:
                            self.workState = 2
                        self.pathTag += 1
                        self.pickOrDeliver = False
                        self.targetPosMid = self.pickupPath[self.pathTag]
                        self.movement_calculation(self.targetPosMid)

                else:
                    # if robot is allocated to a mrt
                    # after the robot reach pickup points, it needs to wait for other robots coming
                    self.wait = True
            else:
                # update robot's position
                self.continuousPos = list(self.continuousPos)
                self.continuousPos[0] += self.deltaX
                self.continuousPos[1] += self.deltaY
                dx, dy = self.model.convert_continuous_to_discrete(self.continuousPos)
                self.model.grid.move_agent(self, (dx, dy))

        # if robot is on its way to delivery point
        else:
            # if robots can reach target pos in this tick
            # robot finish the ongoing task in this tick
            if math.dist(self.continuousPos, self.targetPosMid) < self.avgVelocity:

                self.continuousPos = list(self.targetPosMid)
                dx, dy = self.model.convert_continuous_to_discrete(self.targetPosMid)
                self.model.grid.move_agent(self, (dx, dy))

                index, taskDoneList = self.task_to_index(self.ongoingTask, self.targetPosMid, self.pickOrDeliver)
                # update the task state
                for i in range(len(index)):
                    self.model.schedule._agents[index[i]].done = True
                    # delete corresponding task from task list
                    task = taskDoneList[i]
                    depot = task.depotName
                    num = 0
                    # avoid repeated delete corresponding task from task list
                    if self.currentPartners:
                        for partner in self.currentPartners:
                            if self.unique_id < partner:
                                num += 1
                            else:
                                break
                        if num == len(self.currentPartners):
                            self.model.taskList[depot].pop(self.model.taskList[depot].index(task))
                    else:
                        self.model.taskList[depot].pop(self.model.taskList[depot].index(task))

                if self.ongoingTask[0].taskType == 0:
                    # if robots have already delivered all the items
                    if self.pathTag == len(self.deliveryPath) - 1:
                        self.ongoingTask = []
                        self.workState = 0
                    else:
                        self.pathTag += 1
                        self.pickOrDeliver = True
                        self.targetPosMid = self.deliveryPath[self.pathTag]
                        self.movement_calculation(self.targetPosMid)
                else:
                    # ongoing task is MRT
                    self.ongoingTask = []
                    self.workState = 0
                    for partner in self.currentPartners:
                        agent = self.model.schedule._agents[partner]
                        if math.dist(agent.continuousPos, self.targetPosMid) > 2 * self.avgVelocity:
                            print("error")
                            print("I'm robot " + str(self.unique_id))
                            print("my partner is " + str(agent.unique_id))
                            print("his ongoing task is " + str(agent.ongoingTask))
                            print("his future task is " + str(agent.nextTask))

            else:
                # update robot's position
                self.continuousPos = list(self.continuousPos)
                self.continuousPos[0] += self.deltaX
                self.continuousPos[1] += self.deltaY
                dx, dy = self.model.convert_continuous_to_discrete(self.continuousPos)
                self.model.grid.move_agent(self, (dx, dy))

    def movement_calculation(self, targetPos: list):
        # compute the movement direction and distance for current target position
        if targetPos[1] != self.continuousPos[1] and targetPos[0] != self.continuousPos[0]:
            self.angle = (targetPos[0] - self.continuousPos[0]) / (targetPos[1] - self.continuousPos[1])
            self.deltaY = self.avgVelocity / math.sqrt(1 + math.pow(self.angle, 2)) * \
                          math.copysign(1, (targetPos[1] - self.continuousPos[1]))
            self.deltaX = self.deltaY * self.angle
        elif targetPos[1] == self.continuousPos[1] and targetPos[0] == self.continuousPos[0]:
            self.deltaY = 0
            self.deltaX = 0
        elif targetPos[1] == self.continuousPos[1]:
            self.deltaY = 0
            self.deltaX = self.avgVelocity * math.copysign(1, (targetPos[0] - self.continuousPos[0]))
        elif targetPos[0] == self.continuousPos[0]:
            self.deltaY = self.avgVelocity * math.copysign(1, (targetPos[1] - self.continuousPos[1]))
            self.deltaX = 0

    def find_index(self, task):
        for i in range(self.model.numAgents + self.model.numShipPort, self.model.current_id):
            if task == self.model.schedule._agents[i]:
                return i

    def check_available_tasks(self):
        availableSrt = []
        availableMrt = []
        newTask = False
        currentTime = self.model.schedule.time
        numDepot = self.model.numShipPort
        for i in range(numDepot):
            if self.model.taskList[i]:
                # check the first id of the task list
                truckName = self.model.taskList[i][0].truckID
                for task in self.model.taskList[i]:
                    if task.truckID == truckName:
                        if currentTime == task.startTime:
                            newTask = True
                        if currentTime >= task.startTime:
                            # update tasks' state
                            index = self.find_index(task)
                            self.model.schedule._agents[index].appear = True
                            if not task.selected:
                                # include these tasks in available Srt list and Mrt list
                                if task.taskType == 0:
                                    availableSrt.append(task)
                                else:
                                    availableMrt.append(task)
                    else:
                        break

        return availableSrt, availableMrt, newTask

    def modify_task_schedule(self, task: Task, depotName: int, truckName: int):

        # calculate the task delay rate
        if self.availableTime > task.endTime:
            if task.taskType == 0:
                self.model.srtDelayRate += 1 / self.model.sumSrt
                self.model.taskDelayRate += 1 / (self.model.sumSrt + self.model.sumMrt * 2.5)
            else:
                self.model.mrtDelayRate += 1 / self.model.sumMrt
                self.model.taskDelayRate += 2.5 / (self.model.sumSrt + self.model.sumMrt * 2.5)

        # truck can wait until all the tasks on the truck is finished
        # update the real-time schedule
        if self.model.numTaskList[depotName][truckName] == 0:
            # if it is the last task on that truck
            completionTime = self.availableTime
            # update the real time truck schedule
            self.model.realTimeSchedule[depotName][truckName][1] = completionTime
            # whether the task is delayed
            if completionTime > task.endTime:
                # compute the punishment based on the completion time
                self.model.truckDelayRate += 1 / (self.model.numShipPort * self.model.numTruck)
                self.model.truckDelayTime += (completionTime - task.endTime)
                # if it is not the last truck
                if truckName + 1 < self.model.numTruck:
                    if completionTime > self.model.truckScheduleList[depotName][truckName + 1][0]:
                        # update the real time truck schedule
                        self.model.realTimeSchedule[depotName][truckName + 1][0] = completionTime
                        # update task's starting time
                        for i in range(len(self.model.taskList[depotName])):
                            if self.model.taskList[depotName][i].truckID == truckName + 1:
                                if self.model.taskList[depotName][i].startTime < completionTime:
                                    self.model.taskList[depotName][i].startTime = completionTime
                            else:
                                break

    def find_agents_time(self, number: int):
        # find a number of agents whose available time is similar
        availableTimeList = []
        for i in range(self.model.numAgents):
            agent = self.model.schedule._agents[i]
            availableTimeList.append(agent.availableTime)
        availableTimeList = np.array(availableTimeList) - self.availableTime
        # from minimum to maximum, bubble sort
        availableTimeList, chosenAgentList = self.bubble_sort(availableTimeList, list(range(self.model.numAgents)))
        chosenAgentList.pop(chosenAgentList.index(self.unique_id))
        chosenAgentList.insert(0, self.unique_id)
        chosenAgentList = chosenAgentList[:number]
        return chosenAgentList

    def find_agents_position(self, number: int, agentList: list):
        initialPos = self.availablePos
        agentPosList = [self]
        # agentList stores the index of the agents
        for a in agentList:
            agent = self.model.schedule._agents[a]
            if agent.availablePos != initialPos:
                agentPosList.append(agent)
            if len(agentPosList) == number:
                break
        return agentPosList

    def task_allocation(self, availableSrt: list, availableMrt: list):
        availableTimeList = []
        numAgents = self.model.numRobotsPredict
        # if mrt selection is True, then the best MRT combination contains robot itself
        mrtSelection = False
        # profit for doing srt bundle list
        srtProfit = -100
        # find the robot index list which will be considered
        consideredRobotList = self.find_agents_time(numAgents)
        # Matrix to record potential profit for Mrt
        if availableMrt:
            potentialProfitMrt = [[0] * len(availableMrt) for _ in range(numAgents)]
            # potential profit for Mrt considers the significance of task and the distance from available position to
            # the task's pickup point
            for i in range(numAgents):
                agentId = consideredRobotList[i]
                for j in range(len(availableMrt)):
                    potentialProfitMrt[i][j] = self.potential_profit_mrt(availableMrt[j], agentId)

        # Srt allocation
        if availableSrt:
            if len(availableSrt) > numAgents * 1 or len(availableSrt) == 0:
                agentList = [self.model.schedule._agents[a] for a in consideredRobotList]
            else:
                numSpecAgent = math.ceil(len(availableSrt) / 2)
                agentList = self.find_agents_position(numSpecAgent, consideredRobotList)
            taskBundleList, selectDistanceList = self.srt_allocation(agentList, availableSrt)
            selfIndex = agentList.index(self)

        # store the mrt with the highest profit
        bestTask = None
        # mrt allocation
        bestProfitMrt = -100
        bestMrtProfit = -100
        # contains all the members in the coalition
        bestPartners = []
        bestArriveTime = 0
        bestSumWaitTime = 0

        if availableMrt:
            agentListMrt = consideredRobotList
            # Get the Mrt with the highest profit, and best partners
            for i in range(len(availableMrt)):
                task = availableMrt[i]
                # iterate for every available Mrt
                potentialProfitMrtColumn = [row[i] for row in potentialProfitMrt]
                profit, partners, arriveTime, mrtProfit, sumWaitTime = self.mrt_allocation(task,
                                                                                           potentialProfitMrtColumn,
                                                                                           self.unique_id, agentListMrt,
                                                                                           False, consideredRobotList)
                if profit > bestProfitMrt:
                    bestTask = task
                    bestProfitMrt = profit
                    bestPartners = partners
                    bestArriveTime = arriveTime
                    bestMrtProfit = mrtProfit
                    bestSumWaitTime = sumWaitTime

            # check whether the best combination contains robot itself
            for partner in bestPartners:
                availableTimeList.append(self.model.schedule._agents[partner].availableTime)
                agentListMrt.pop(agentListMrt.index(partner))
                if partner == self.unique_id:
                    mrtSelection = True

        # 机器人在上次MRT allocation时没有被分配到MRT，等待下一个合适的时间节点，即推测的最优秀的组合中最早的availble time,或者当有新的srt或者MRT加入时。
        # compute the srt profit according to the coalition members
        if availableSrt:
            if mrtSelection:
                srtProfit = 0
                partnerList = []
                for partner in bestPartners:
                    # find the corresponding index of partner in the agentList
                    agent = self.model.schedule._agents[partner]
                    for i in range(len(agentList)):
                        if agentList[i] == agent:
                            totalSignificance = 0
                            for task in taskBundleList[i]:
                                totalSignificance += self.significance(task) * task.weight / self.payload
                            time = selectDistanceList[i] / self.avgVelocity
                            # time is for the whole trip
                            srtProfit += (totalSignificance - time / self.model.timeThreshold)
                            partnerList.append(agent)
                            break
                # recompute the MRT profit when the agent list does not contain all the members in the best partner
                if len(partnerList) != len(bestPartners):
                    bestProfitMrt = 0
                    # recalculate the best mrt profit
                    for b in partnerList:
                        # calculate distance, wait time and significance
                        duration = math.dist(b.availablePos, bestTask.pickPoint) / self.avgVelocity
                        waitTime = (bestArriveTime - b.availableTime - duration) / (
                                self.model.timeThreshold / self.model.waitThresholdFactor)
                        if waitTime < 0:
                            print("error")
                        duration /= (self.model.timeThreshold / self.model.mrtDistanceFactor)
                        bestProfitMrt += self.significance(
                            bestTask) * self.model.mrtWeightBalanceFactor - duration - waitTime
            else:
                srtProfit = 100
        else:
            bestProfitMrt = 100

        if mrtSelection:
            if bestProfitMrt > srtProfit:
                if bestTask.numRobotsNeeded != len(bestPartners):
                    print("error")
                # record the sum of wait Time
                self.model.totalWaitTime += bestSumWaitTime
                bestPartners = list(bestPartners)
                bestPartners.pop(bestPartners.index(self.unique_id))
                bestPartners.insert(0, self.unique_id)
                # ongoing task is mrt
                i = 0
                for partner in bestPartners:
                    set1 = bestPartners.copy()
                    set1.pop(i)
                    if partner == self.unique_id:
                        # update the ongoing Task for robot itself
                        self.ongoingTask = []
                        self.ongoingTask.append(bestTask)
                        # update the current partners
                        self.currentPartners = set1
                        # update the available time and position and working state
                        self.update_agent_state(bestArriveTime, 0)

                    else:
                        self.model.travelDistanceList[partner] += (
                                math.dist(self.availablePos, bestTask.pickPoint)
                                + math.dist(bestTask.pickPoint,
                                            bestTask.deliveryPoint))
                        # update the next task
                        self.model.schedule._agents[partner].nextTask.append(bestTask)
                        # update the future partners
                        self.model.schedule._agents[partner].futurePartners.append(set1)
                        # update the available time and pos
                        self.model.schedule._agents[partner].availableTime = self.availableTime
                        # update the available time and pos
                        self.model.schedule._agents[partner].availablePos = self.availablePos
                    i += 1
        else:
            if availableSrt:
                self.ongoingTask = taskBundleList[selfIndex]
                self.update_agent_state(0, selectDistanceList[selfIndex])
                self.model.travelDistanceList[self.unique_id] += selectDistanceList[selfIndex]
        if availableTimeList:
            if not availableSrt and not mrtSelection:
                self.runTime = min(availableTimeList)
                print("I'm robot " + str(self.unique_id), "I will make the next decision at " + str(self.runTime))

    def srt_allocation(self, agentList: list, availableSrt: list):

        availableSrt1 = availableSrt.copy()
        # agentList is the list of participated robots
        numAgents = len(agentList)
        # find the index of robots itself in the agentList
        selfIndex = agentList.index(self)
        self.ongoingTask = []
        self.pickupPath = []
        self.deliveryPath = []
        # Task allocation among SRT
        # store the remaining capacity for robots participating in srt allocation
        remainingCapacityList = [agent.payload for agent in agentList]
        # store the selected Srt bundle for robots
        taskBundleList = [[] for _ in range(numAgents)]
        # the number of participated robots list, store the unique id of the robots
        participatedRobotList = list(range(numAgents))
        # distanceList store the distance of selected path
        selectDistanceList = [0] * numAgents
        # conduct single-robot task allocation
        while availableSrt1 and len(participatedRobotList) > 0:
            # only need to store the path for robot itself
            pickupPathList = []
            deliveryPathList = []
            # store the distance for every robot and task
            distanceList = [[0] * len(availableSrt1) for _ in range(len(participatedRobotList))]
            # selected Srt in this round
            selectedSrtList = []
            # this is a marginal profit for every task
            potentialProfitSrt = [[0] * len(availableSrt1) for _ in range(len(participatedRobotList))]
            # The potential profit for Srt considers the significance of the task (remaining time and weight) and
            # additional distance
            for i in range(len(participatedRobotList)):
                # calculate the profit for every available SRT
                for j in range(len(availableSrt1)):
                    index = participatedRobotList[i]
                    agentID = agentList[index].unique_id
                    # if remaining capacity is enough
                    if availableSrt1[j].weight <= remainingCapacityList[index]:
                        potentialProfitSrt[i][j], distanceList[i][j], a, b = self.potential_profit_srt(
                            availableSrt1[j], agentID,
                            taskBundleList[index],
                            selectDistanceList[index], remainingCapacityList[index])
                        if agentID == self.unique_id:
                            # encouragement factor
                            potentialProfitSrt[i][j] *= self.model.encouragementFactor
                        if selectDistanceList[index] > distanceList[i][j]:
                            print("error")
                    else:
                        potentialProfitSrt[i][j] = 0
                        a = []
                        b = []
                    if agentID == self.unique_id:
                        # update the pickup path and delivery path of robot
                        pickupPathList.append(a)
                        deliveryPathList.append(b)

            # through srt allocation, robots will possibly be allocated an SRT
            sol = self.maximum_match(availableSrt1, potentialProfitSrt, participatedRobotList)
            # update the characters of the participated robots
            for i in range(len(participatedRobotList)):
                if i in sol:
                    index = participatedRobotList[i]
                    agentID = agentList[index].unique_id

                    # update the profit for Srt bundle
                    taskIndex = sol[i]
                    weight = availableSrt1[taskIndex].weight
                    if remainingCapacityList[index] - weight >= 0:

                        # reduce the remaining capacity
                        remainingCapacityList[index] -= weight
                        # update task bundle list of robot i
                        taskBundleList[index].append(availableSrt1[taskIndex])
                        selectDistanceList[index] = distanceList[i][taskIndex]
                        # store selected srt, cuz the index will change while deleting tasks from available task list
                        selectedSrtList.append(availableSrt1[taskIndex])
                        if agentID == self.unique_id:
                            self.pickupPath = pickupPathList[taskIndex]
                            self.deliveryPath = deliveryPathList[taskIndex]
                            if len(taskBundleList[index]) == 1:
                                if self.pickupPath[0] != taskBundleList[index][0].pickPoint:
                                    print("error")
                    else:
                        print("error")
                        sol = self.maximum_match(availableSrt1, potentialProfitSrt, participatedRobotList)
            # delete selected tasks from available srt list in this round
            for task in selectedSrtList:
                index = availableSrt1.index(task)
                availableSrt1.pop(index)

            participatedRobotList = []
            # update the participatedRobotList
            for i in range(numAgents):
                if remainingCapacityList == 0:
                    break
                for task in availableSrt1:
                    if task.weight <= remainingCapacityList[i]:
                        participatedRobotList.append(i)
                        break
        return taskBundleList, selectDistanceList

    def maximum_match(self, availableSrt: list, profitListSrt: list, participateRobots: list):
        interestedTaskList = []
        for i in range(len(availableSrt)):
            column = [row[i] for row in profitListSrt]
            # the elements in the column cannot all be 0
            for element in column:
                if element != 0:
                    # interestedTaskList stores the index of tasks which robots are interested in
                    interestedTaskList.append(i)
                    break
        # construct the bipartite graph
        left = list(range(len(participateRobots)))  # robot nodes
        right = list(
            range(len(participateRobots), len(participateRobots) + len(interestedTaskList)))  # available task nodes
        B = nx.Graph()
        B.add_nodes_from(left, bipartite=0)
        B.add_nodes_from(right, bipartite=1)
        for i in range(len(participateRobots)):
            for j in range(len(interestedTaskList)):
                index = interestedTaskList[j]
                if profitListSrt[i][index] != 0:
                    B.add_edge(left[i], right[j],
                               weight=profitListSrt[i][index] + 1)  # edit the edge weight of bipartite graph
        sol = self.maximum_weight_full_matching(B, left)  # max weight matching
        for i in range(len(participateRobots)):
            if i in sol:
                index = sol[i] - len(participateRobots)
                sol[i] = interestedTaskList[index]
                if profitListSrt[i][sol[i]] == 0:
                    sol.pop(i)
        return sol

    def mrt_allocation(self, task: Task, potentialProfitMrt: list, agentID: int, agentList: list, reallocation: bool,
                       consideredRobotList: list):
        # Initialisation
        numRobotsNeeded = task.numRobotsNeeded
        combination = list(itertools.combinations(agentList, numRobotsNeeded))
        greatCoalition = []
        greatArriveTime = 0
        greatProfit = -1000
        greatSumWaitTime = 0
        bestDuration = 0
        for permutation in combination:
            if reallocation:
                if agentID not in permutation:
                    continue
            arriveTime, sumWaitTime, duration = self.arrive_time_calculation_mrt(permutation, task)
            currentProfit = self.profit_calculation_mrt(permutation, sumWaitTime, potentialProfitMrt,
                                                        consideredRobotList)
            if currentProfit > greatProfit:
                # update the coalition set, arrive time and the profit
                greatCoalition = permutation
                greatArriveTime = arriveTime
                greatProfit = currentProfit
                bestDuration = duration
                greatSumWaitTime = sumWaitTime

        mrtProfit = self.significance(
            task) * numRobotsNeeded * self.model.mrtWeightBalanceFactor - bestDuration / self.model.timeThreshold

        return greatProfit, greatCoalition, greatArriveTime, mrtProfit, greatSumWaitTime

    def significance(self, task: Task):
        remainTime = task.endTime - self.model.schedule.time
        threshold = self.model.timeThreshold * self.model.timeFactor
        if remainTime > threshold:
            stimulusFactor = self.model.stimulusFactor
            return task.taskPriority * stimulusFactor
        else:
            stimulusFactor = self.model.stimulusFactor + \
                             (math.exp((threshold - remainTime) / self.model.timeThreshold) / math.exp(1)) * (
                                     self.model.maxStimulusFactor - self.model.stimulusFactor)
            return stimulusFactor * task.taskPriority

    def weight_srt(self, task: Task, remainingCapacity: int):

        weight = task.weight / remainingCapacity
        return weight

    def profit_calculation_mrt(self, coalition: list, sumWaitTime: int, potentialProfitMrt: list,
                               consideredRobotList: list):
        # consider the waiting time, and robots' Srt profit
        profit: float = 0
        # calculate the sum of potential MRT profit, waiting time
        for agent in coalition:
            # potentialProfitMrt = significance(task) - time(pick point, available pos)
            index = consideredRobotList.index(agent)
            profit += potentialProfitMrt[index]
        timeThreshold = self.model.timeThreshold / self.model.waitThresholdFactor
        profit -= (sumWaitTime / timeThreshold) * self.model.waitFactor
        return profit

    def switch(self, set1: list, set2: list, pos1: int, pos2: int):
        a = set1[pos1]
        b = set2[pos2]
        set1.pop(pos1)
        set1.insert(pos1, b)
        set2.pop(pos2)
        set2.insert(pos2, a)

    def potential_profit_srt(self, task: Task, agentId: int, selectedTaskList: list, previousDistance: float,
                             remainingCapacity: int):

        agent = self.model.schedule._agents[agentId]
        # pickup and delivery position list
        pickupList = []
        deliveryList = []
        if selectedTaskList:
            pickupList = [task1.pickPoint for task1 in selectedTaskList]
            deliveryList = [task1.deliveryPoint for task1 in selectedTaskList]
        # list storing the position of pickup and delivery point
        pickupList.append(task.pickPoint)
        deliveryList.append(task.deliveryPoint)

        # get rid of the duplicates from the list
        pickupList.sort()
        pickupList = list(pickupList for pickupList, _ in itertools.groupby(pickupList))
        deliveryList.sort()
        deliveryList = list(deliveryList for deliveryList, _ in itertools.groupby(deliveryList))
        distance = 0
        nodeCount = 1 + len(pickupList) + len(deliveryList)

        if len(pickupList) >= 2 or len(deliveryList) >= 2:
            # distance, solution = SA.sa_path(nodeCount, agent.availablePos,
            #                                 pickupList, deliveryList)
            distance, bestPickupList, bestDeliveryList = self.calculate_shortest_distance(agent, pickupList, deliveryList)
            timeThreshold = self.model.timeThreshold / (len(pickupList) * self.model.srtThresholdFactor)
        else:

            distance += math.dist(agent.availablePos, pickupList[0])
            distance += math.dist(pickupList[0], deliveryList[0])
            # solution = [0, 1, 2]
            bestPickupList = [pickupList[0]]
            bestDeliveryList = [deliveryList[0]]
            timeThreshold = self.model.timeThreshold
        significance = self.significance(task)
        weight = self.weight_srt(task, remainingCapacity)
        # this profit is used for srt allocation
        time = (distance - previousDistance) / self.avgVelocity
        if distance - previousDistance < 0:
            print("error")
        profit = significance * weight - time / timeThreshold

        return profit, distance, bestPickupList, bestDeliveryList

    def calculate_shortest_distance(self, agent, pickupList: list, deliveryList: list):
        lengthPickup = len(pickupList)
        lengthDelivery = len(deliveryList)
        # index list permutation
        permutationPick = list(itertools.permutations(range(lengthPickup), lengthPickup))
        permutationDeliver = list(itertools.permutations(range(lengthDelivery), lengthDelivery))
        bestPermutationPick = []
        bestPermutationDeliver = []
        bestDistance = 1000000
        for i in range(len(permutationPick)):
            for j in range(len(permutationDeliver)):
                permutationPick1 = list(permutationPick[i])
                permutationDeliver1 = list(permutationDeliver[j])
                distance = self.calculate_distance(agent, permutationPick1, permutationDeliver1, pickupList, deliveryList)
                if distance < bestDistance:
                    # best permutation pick and deliver contains position list
                    bestPermutationPick = [pickupList[a] for a in permutationPick1]
                    bestPermutationDeliver = [deliveryList[b] for b in permutationDeliver1]
                    bestDistance = distance
        return bestDistance, bestPermutationPick, bestPermutationDeliver

    def calculate_distance(self, agent, permutationPick: list, permutationDeliver: list, pickupList: list, deliveryList: list):
        distance = math.dist(agent.availablePos, pickupList[permutationPick[0]])
        for i in range(1, len(permutationPick)):
            distance += math.dist(pickupList[permutationPick[i - 1]], pickupList[permutationPick[i]])
        distance += math.dist(pickupList[permutationPick[-1]], deliveryList[permutationDeliver[0]])
        for i in range(1, len(permutationDeliver)):
            distance += math.dist(deliveryList[permutationDeliver[i - 1]], deliveryList[permutationDeliver[i]])
        return distance

    def potential_profit_mrt(self, task: Task, agentId: int) -> float:
        # without considering the waiting time
        agent = self.model.schedule._agents[agentId]
        time = math.dist(agent.availablePos, task.pickPoint) / self.avgVelocity
        # the significance of mrt is affected by remaining time of it
        significance = self.significance(task)
        timeThreshold = self.model.timeThreshold / self.model.mrtDistanceFactor
        profit = significance * self.model.mrtWeightBalanceFactor - time / timeThreshold * (1 - self.model.waitFactor)
        return profit

    def update_agent_state(self, bestArriveTime: int, distance: float):

        velocity = self.avgVelocity
        ongoingTask = self.ongoingTask
        if not ongoingTask:
            # if the next action is empty
            self.state = 0  # idle mode
            self.currentPartners = []
            self.deliveryPath = []
            self.pickupPath = []

        else:
            # if the next action is a Srt
            if ongoingTask[0].taskType == 0:
                self.currentPartners = []
                # update the available time for the robot
                self.availableTime += math.ceil(distance / velocity)
                # update the target position of robot
                if not self.deliveryPath:
                    print("error")
                self.availablePos = self.deliveryPath[-1]
                # update the working state of robot
                self.state = 1
                self.targetPosMid = self.pickupPath[0]
                # update the movement angle and distance of ongoing task
                self.movement_calculation(self.targetPosMid)
            # if the next action is Mrt
            elif ongoingTask[0].taskType == 1:
                self.model.travelDistanceList[self.unique_id] += (math.dist(self.availablePos, ongoingTask[0].pickPoint)
                                                                  + math.dist(ongoingTask[0].pickPoint,
                                                                              ongoingTask[0].deliveryPoint))
                # compute the available time for robot itself
                self.availableTime = math.ceil(
                    bestArriveTime + math.dist(ongoingTask[0].pickPoint, ongoingTask[0].deliveryPoint) / velocity)
                # compute the available position for robot itself
                self.availablePos = ongoingTask[0].deliveryPoint
                self.state = 1
                self.pickupPath = []
                self.deliveryPath = []
                self.pickupPath.append(ongoingTask[0].pickPoint)
                self.deliveryPath.append(ongoingTask[0].deliveryPoint)
                self.targetPosMid = self.pickupPath[0]
                self.movement_calculation(self.targetPosMid)

    def arrive_time_calculation_mrt(self, partners: list, task: Task):
        arriveTime = []
        availableTimeList = []
        agents = self.model.schedule._agents
        for i in range(len(partners)):
            agent = agents[partners[i]]
            travelDistance = math.dist(agent.availablePos, task.pickPoint)
            availableTimeList.append(agent.availableTime)
            arriveTime1 = agent.availableTime + math.ceil(travelDistance / agent.avgVelocity)
            arriveTime.append(arriveTime1)
        sumWaitTime = sum(max(arriveTime) - np.array(arriveTime))
        # duration is the total time cost of all partners
        duration = sum(np.array(arriveTime) - np.array(availableTimeList)) + \
                   len(partners) * math.dist(task.pickPoint, task.deliveryPoint)
        return max(arriveTime), sumWaitTime, duration

    def bubble_sort(self, nums: list, Index: list):
        # from minimum to maximum
        # We set swapped to True so the loop looks runs at least once
        swapped = True
        length = len(nums)
        while swapped:
            swapped = False
            for i in range(length - 1):
                if nums[i] > nums[i + 1]:
                    # Swap the elements
                    nums[i], nums[i + 1] = nums[i + 1], nums[i]
                    Index[i], Index[i + 1] = Index[i + 1], Index[i]
                    # Set the flag to True so we'll loop again
                    swapped = True
            length -= 1
        return nums, Index

    def maximum_weight_full_matching(self, G, top_nodes=None, weight='weight'):
        r"""Returns the maximum weight full matching of the bipartite graph `G`.
        Let :math:`G = ((U, V), E)` be a complete weighted bipartite graph with
        real weights :math:`w : E \to \mathbb{R}`. This function then produces
        a maximum matching :math:`M \subseteq E` which, since the graph is
        assumed to be complete, has cardinality
        .. math::
           \lvert M \rvert = \min(\lvert U \rvert, \lvert V \rvert),
        and which minimizes the sum of the weights of the edges included in the
        matching, :math:`\sum_{e \in M} w(e)`.
        When :math:`\lvert U \rvert = \lvert V \rvert`, this is commonly
        referred to as a perfect matching; here, since we allow
        :math:`\lvert U \rvert` and :math:`\lvert V \rvert` to differ, we
        follow Karp [1]_ and refer to the matching as *full*.
        Parameters
        ----------
        G : NetworkX graph
          Undirected bipartite graph
        top_nodes : container
          Container with all nodes in one bipartite node set. If not supplied
          it will be computed.
        weight : string, optional (default='weight')
           The edge data key used to provide each value in the matrix.
        Returns
        -------
        matches : dictionary
          The matching is returned as a dictionary, `matches`, such that
          ``matches[v] == w`` if node `v` is matched to node `w`. Unmatched
          nodes do not occur as a key in matches.
        Raises
        ------
        ValueError : Exception
          Raised if the input bipartite graph is not complete.
        ImportError : Exception
          Raised if SciPy is not available.
        Notes
        -----
        The problem of determining a minimum weight full matching is also known as
        the rectangular linear assignment problem. This implementation defers the
        calculation of the assignment to SciPy.
        References
        ----------
        .. [1] Richard Manning Karp:
           An algorithm to Solve the m x n Assignment Problem in Expected Time
           O(mn log n).
           Networks, 10(2):143–152, 1980.
        """
        try:
            import scipy.optimize
        except ImportError:
            raise ImportError('minimum_weight_full_matching requires SciPy: ' +
                              'https://scipy.org/')
        left = set(top_nodes)
        right = set(G) - left
        # Ensure that the graph is complete. This is currently a requirement in
        # the underlying  optimization algorithm from SciPy, but the constraint
        # will be removed in SciPy 1.4.0, at which point it can also be removed
        # here.
        # for (u, v) in itertools.product(left, right):
        # As the graph is undirected, make sure to check for edges in
        # both directions
        #    if (u, v) not in G.edges() and (v, u) not in G.edges():
        #        raise ValueError('The bipartite graph must be complete.')
        U = list(left)
        V = list(right)
        weights = biadjacency_matrix(G, row_order=U,
                                     column_order=V, weight=weight).toarray()
        left_matches = scipy.optimize.linear_sum_assignment(-weights)
        d = {U[u]: V[v] for u, v in zip(*left_matches)}
        # d will contain the matching from edges in left to right; we need to
        # add the ones from right to left as well.
        # d.update({v: u for u, v in d.items()})
        return d
