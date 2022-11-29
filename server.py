import math

from AgentModelNew import Robots
from Task import Task
from ShippingPort import ShippingPort
import mesa
from ProposedModel import Model


def agent_portrayal(agent):
    if agent is None:
        return

    portrayal = {}

    if type(agent) is Robots:
        if agent.workState == 0:
            portrayal["Shape"] = "robot01.jpg"
        elif agent.workState == 1:
            portrayal["Shape"] = "SRT01.jpg"
        elif agent.workState == 2:
            portrayal["Shape"] = "MTR01.jpg"
        elif agent.workState == 3:
            portrayal["Shape"] = "MRT01.jpg"

        portrayal["Filled"] = "true"
        portrayal["Layer"] = 2
        portrayal["r"] = 0.6
        portrayal["Color"] = "red"

    elif type(agent) is ShippingPort:
        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["w"] = 1
        portrayal["h"] = 1
        portrayal["Color"] = "blue"
        portrayal["text"] = round(agent.numRemainingTask, 1)
        portrayal["text_color"] = "White"

    elif type(agent) is Task:
        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 1
        portrayal["w"] = 1
        portrayal["h"] = 1
        if 0 < agent.remainingTime < 3 * agent.model.timeThreshold:
            color = math.floor(agent.remainingTime / (3 * agent.model.timeThreshold) * 255)
            colorStr = hex(color)
            if len(colorStr) == 3:
                string = '0' + colorStr[2:]
            else:
                string = colorStr[2:]
            portrayal["Color"] = "#ff" + string + "00"
            portrayal["text_color"] = "Black"
        elif agent.remainingTime < 0:
            portrayal["Color"] = "#ff0000"
            portrayal["text_color"] = "Black"
        else:
            portrayal["Color"] = "	#40ff00"
            portrayal["text_color"] = "Black"
        portrayal["text"] = round(agent.weight, 1)

    return portrayal

# N: int, limitX: list, limitY: list, numTruck: int, numSrt: list, numMrt: list,
#                  agentPayload: int, avgVelocity: float, numRobotsNeeded: int,
#                  timeFactor: float, waitFactor: float, stimulusFactor: float,
#                  mrtDistanceFactor: float, mrtWeightBalanceFactor: float, waitThresholdFactor: float,
#                  maxStimulusFactor: float, encouragementFactor: float,
#                  bundleParameter: float, numRobotsSpeculated: int, numKeepMRT: float,
#                  numShipPort: int, weightRangeSrt: list, requiredRobots: int, randomSeed: int, emergencyProportion: list
para = {
    "N": 10,
    "limitX": [0, 300],
    "limitY": [0, 100],
    "numTruck": 3,
    "numSrt": [10, 20],
    "numMrt": [10, 20],
    "agentPayload": 50,
    "avgVelocity": 1,
    "numRobotsNeeded": 3,

    "timeFactor": 2,
    "waitFactor": 0.8,
    "stimulusFactor": 0.8,
    "mrtDistanceFactor": 2.5,
    "mrtWeightBalanceFactor": 0.9,
    "waitThresholdFactor": 2.5,
    "maxStimulusFactor": 2,



    'encouragementFactor': 1.1,
    "bundleParameter": 0.8,
    'numRobotsSpeculated': 6,
    "numKeepMRT": 0.1,

    "numShipPort": 3,
    "weightRangeSrt": [0, 50],
    'requiredRobots': 12,
    'randomSeed': 5,
    'emergencyProportion': [0.4, 0.4]
}
grid = mesa.visualization.CanvasGrid(agent_portrayal, 60, 20, 1500, 500)
chart_element = mesa.visualization.ChartModule(
    [
        {"Label": "loading Point 1", "Color": "#AA0000"},
        {"Label": "loading Point 2", "Color": "#666666"},
        {"Label": "loading Point 3", "Color": "#00AA00"},
    ],
    data_collector_name='datacollector'
)
# server = mesa.visualization.ModularServer(
#     Model, [grid, chart_element], "Proposed Model", para
# )
server = mesa.visualization.ModularServer(
    Model, [grid], "Proposed Model", para
)
server.port = 8521  # The default
server.launch()
