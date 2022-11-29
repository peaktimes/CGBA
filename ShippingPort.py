import mesa
import math
import random
from Task import Task
import numpy as np
import networkx as nx
from typing import List
import copy


class ShippingPort(mesa.Agent):

    def __init__(self, unique_id, model, cPos: list, numTruck: int, truckName: int, numRemainingTask: int,
                 portIndex: int):
        super().__init__(unique_id, model)
        self.cPos = cPos
        self.numTruck = numTruck
        self.truckName = truckName
        self.numRemainingTask = numRemainingTask
        self.portIndex = portIndex


    def calculate_remaining_task(self):
        self.numRemainingTask = 0
        currentTime = self.model.schedule.time
        if self.model.taskList[self.portIndex]:
            truckName = self.model.taskList[self.portIndex][0].truckID
            for task in self.model.taskList[self.portIndex]:
                if task.truckID == truckName:
                    self.truckName = truckName
                    if currentTime >= task.startTime:
                        self.numRemainingTask += 1
                else:
                    break
        return self.numRemainingTask

    def step(self):
        self.calculate_remaining_task()
