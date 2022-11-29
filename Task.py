import mesa


class Task(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.taskType: bool = 0  # 0 is SRT, 1 is MRT
        self.loadOrUnload: bool = 0  # 0 is loading task, 1 is unloading task
        self.taskID: int = 0
        self.truckID: int = 0
        self.depotName: int = 0
        self.taskPriority: int = 1  # task reward
        self.startTime: float = 0  # task start time (sec)
        self.endTime: float = 0  # task expiry time (sec)
        self.pickPoint: list = [0, 0]  # the continuous position of task picking point
        self.deliveryPoint: list = [0, 0]  # the continuous position of task delivery point
        self.weight: int = 0  # the weight of item
        self.weightScale: list = [0, 50]  # weight scale of single robot task
        self.numRobotsNeeded: int = 1
        self.appear: bool = False  # whether the task should appear on the map
        self.picked: bool = False  # whether the task has been picked by the robot
        self.done: bool = False  # whether the task is finished by robot
        self.selected: bool = False  # whether the task is selected by a robot
        self.dPos: tuple = (0, 0)
        self.remainingTime: int = 0  # the remaining time of the task

    def step(self):
        currentTime = self.model.schedule.time
        self.remainingTime = self.endTime - currentTime
        if self.picked:
            self.appear = False
        if self.done:
            self.appear = False
        if not self.appear:
            if self.pos is not None:
                self.model.grid.remove_agent(self)
        else:
            if self.pos is None:
                self.model.grid.place_agent(self, self.dPos)
