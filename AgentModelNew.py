import mesa
import math
import random
from Task import Task
import numpy as np
import networkx as nx
from typing import List
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment
import itertools
import time as MyTime
import pandas as pd

TaskList = List[Task]


class Robots(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.avgVelocity: float = 1  # agent cruise velocity (m/s)
        self.payload: int = 50  # the loading capacity of robots (kg)
        self.prevState: bool = True  # busy:1, idle:0
        self.currentState: bool = False  # busy:1, idle:0

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

    def step(self):

        time = self.model.schedule.time
        agents = self.model.schedule._agents
        ongoingTaskList = [agents[i].ongoingTask for i in range(self.model.numAgents)]
        nextTaskList = [agents[i].nextTask for i in range(self.model.numAgents)]
        availableTimeList = [agents[i].availableTime for i in range(self.model.numAgents)]

        if not self.ongoingTask:
            # when robot finishes its ongoing task, it needs to update its state
            self.pickOrDeliver = False
            self.wait = False
            self.currentState = False
            self.pathTag = 0
            self.pickupPath = []
            self.deliveryPath = []
            self.workState = 0

            if self.nextTask:
                task = self.nextTask[0]
                self.ongoingTask.append(task)
                if task.taskType:
                    selfIndex = self.model.bestCoalition[task].index(self.unique_id)
                    a = self.model.bestCoalition[task]
                    a.pop(selfIndex)
                    self.currentPartners = a
                else:
                    self.currentPartners = []
                self.nextTask.pop(0)
                # 2d matrix
                self.pickupPath.append(self.ongoingTask[0].pickPoint)
                self.deliveryPath.append(self.ongoingTask[0].deliveryPoint)
                self.targetPosMid = self.pickupPath[self.pathTag]
                self.movement_calculation(self.targetPosMid)

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

    def conduct_task_allocation(self):
        # judge whether to conduct task allocation or not
        a = False
        if self.currentState != self.prevState and self.prevState == True:
            a = True
        return a | self.model.taskAllocationToken

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
                    truck = task.truckID
                    num = 0
                    completionTime = self.model.schedule.time
                    # avoid repeated delete corresponding task from task list
                    if self.currentPartners:
                        for partner in self.currentPartners:
                            if self.unique_id < partner:
                                num += 1
                            else:
                                break
                        if num == len(self.currentPartners):
                            self.model.taskList[depot].pop(self.model.taskList[depot].index(task))
                            self.model.numTaskList[depot][truck] -= 1
                            self.calculate_task_delay_rate(completionTime, task)
                            if self.model.numTaskList[depot][truck] == 0:
                                self.modify_task_schedule(task, depot, truck, completionTime)
                    else:
                        self.model.taskList[depot].pop(self.model.taskList[depot].index(task))
                        self.model.numTaskList[depot][truck] -= 1
                        self.calculate_task_delay_rate(completionTime, task)
                        # if this is the last task
                        if self.model.numTaskList[depot][truck] == 0:
                            self.modify_task_schedule(task, depot, truck, self.model.schedule.time)

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
        currentTime = self.model.schedule.time
        numDepot = self.model.numShipPort
        for i in range(numDepot):
            if self.model.taskList[i]:
                # check the first id of the task list
                truckName = self.model.taskList[i][0].truckID
                for task in self.model.taskList[i]:
                    if task.truckID == truckName:
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
                            if currentTime == task.startTime:
                                self.model.taskAllocationToken = True
                    else:
                        break
        return availableSrt, availableMrt

    def calculate_task_delay_rate(self, completionTime: int, task: Task):
        '''
        calculate the task delay rate
        :param completionTime: the completion time of a task
        :param task: the finished task
        :return:
        '''
        # calculate the task delay rate
        if completionTime > task.endTime:
            if task.taskType == 0:
                self.model.srtDelayRate += 1 / self.model.sumSrt
                self.model.taskDelayRate += 1 / (self.model.sumSrt + self.model.sumMrt * 2.5)
            else:
                self.model.mrtDelayRate += 1 / self.model.sumMrt
                self.model.taskDelayRate += 2.5 / (self.model.sumSrt + self.model.sumMrt * 2.5)

    def modify_task_schedule(self, task: Task, depotName: int, truckName: int, completionTime: int):
        '''
        Calculate the truck delay time and modify the information of tasks on the next truck, since the appearing time might change
        :param task: the final task on the truck (depotName, truckName)
        :param depotName: the name of the depot
        :param truckName: the name of the truck
        :param completionTime: the completion time of the task
        :return:
        '''
        if completionTime > task.endTime:
            # update the real time truck schedule
            self.model.realTimeSchedule[depotName][truckName][1] = completionTime
            # whether the task is delayed
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
        availableTimeList, chosenAgentList = self.bubble_sort(availableTimeList,
                                                              list(range(self.model.numAgents)))
        chosenAgentList.pop(chosenAgentList.index(self.unique_id))
        chosenAgentList.insert(0, self.unique_id)
        chosenAgentList = chosenAgentList[:number]
        return chosenAgentList




