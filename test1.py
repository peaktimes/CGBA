from ProposedModel import Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os

dirname = os.getcwd()

if __name__ == "__main__":
    number = 8
    numRandom = 15
    truckDelayTimeList = []

    numRobotsSpeculatedList = list(range(3, 11))

    for i in range(number):
        truckDelayTime = []
        for k in range(numRandom):
            numRobots = 10
            requiredRobots = 12
            limitX = [0, 300]
            limitY = [0, 100]
            numTruck = 3
            numSrt = [40, 40]
            numMrt = [40, 40]
            agentPayload = 50
            avgVelocity = 1
            numRobotsNeeded = 3

            timeFactor = 2
            timeInfluFactor = 0.7
            encouragementFactor = 1.0
            numSRTInfluence = 0.25
            mrtWeightBalanceFactor = 0.8
            waitThresholdFactor = 4

            numRobotsSpeculated = numRobotsSpeculatedList[i]
            numKeepMRT = 0.1

            numShipPort = 3
            weightRangeSrt = [0, 50]
            randomSeed = k
            emergencyProportion = [0, 0]

            iteration = 1

            emptyModel = Model(numRobots, limitX, limitY, numTruck, numSrt, numMrt, agentPayload, avgVelocity,
                               numRobotsNeeded,
                               timeFactor, timeInfluFactor, encouragementFactor, numSRTInfluence,
                               mrtWeightBalanceFactor, waitThresholdFactor,
                               numRobotsSpeculated, numKeepMRT,
                               numShipPort, weightRangeSrt, requiredRobots, randomSeed, emergencyProportion)

            allocatedSRT = []
            payloadUtilization = []

            for j in range(60000):
                emptyModel.step()

            for j in range(numShipPort):
                print("This is the real time truck schedule of ship port " + str(j))
                print(emptyModel.realTimeSchedule[j])
                print("This is the planned truck schedule of ship port " + str(j))
                print(emptyModel.truckScheduleList[j])

            avgTruckDelayTime = emptyModel.truckDelayTime / (3 * numTruck)
            truckDelayTime.append(avgTruckDelayTime)
            # print("This is the truck delay rate: " + str(emptyModel.truckDelayRate * 100))
            print("This is the average of truck delay time: " + str(emptyModel.truckDelayTime / (3 * numTruck)))
            # print("This is the srt delay rate: " + str(emptyModel.srtDelayRate * 100))
            # print("This is the mrt delay rate: " + str(emptyModel.mrtDelayRate * 100))
            # print("This is the task delay rate: " + str(emptyModel.taskDelayRate * 100))
            # print("The traveling distance of robot is: " + str(emptyModel.travelDistanceList))
            # print("The idle time of robots is: " + str(emptyModel.idleTimeList))
            # print("The cumulative computation time is: " + str(sum(emptyModel.computationTime) / numRobots))
            # print("The total waiting time is: " + str(emptyModel.totalWaitTime))
            # print("The length of record time is:" + str(len(emptyModel.recordTime)))
        truckDelayTimeList.append(truckDelayTime)

a = pd.DataFrame(np.array(truckDelayTimeList), index=list(range(3, 11)))
filename = os.path.join(dirname, 'data\\truckDelayTimeList_40srt_40mrt.csv')
a.to_csv(filename)
avgTruckDelayTimeList = [np.average(truckDelayTimeList[i]) for i in range(number)]
fig, ax = plt.subplots()

# make a plot
xPoint = numRobotsSpeculatedList
ax.plot(xPoint, avgTruckDelayTimeList, color="red", linestyle='solid', linewidth=2)
ax.set_xlabel("The number of speculated robots", fontsize=14)
ax.set_ylabel("Average truck delay time (sec)", fontsize=14)
ax.legend()
plt.title('num_SRT=40, numMRT=40')
plt.show()
fig.savefig(
    'truckDelayTimeList_40srt_40mrt.jpg',
    format='jpeg', dpi=600, bbox_inches='tight')
