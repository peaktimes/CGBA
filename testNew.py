from ProposedModel import Model
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    number = 1
    truckDelayTimeList = []
    srtDelayRateList = []
    mrtDelayRateList = []
    truckDelayRateList = []
    travelDistanceList = []
    for i in range(number):
        numRobots = 10
        requiredRobots = 12
        limitX = [0, 300]
        limitY = [0, 100]
        numTruck = 1
        numSrt = [10, 10]
        numMrt = [10, 10]
        agentPayload = 50
        avgVelocity = 1
        numRobotsNeeded = 3

        numAllocatedSRTList = []
        numAllocatedMRTList = []

        # timeFactor = 3
        # timeInfluFactor = 0.7
        # encouragementFactor = 1.0
        # numSRTInfluence = 0.25
        # mrtWeightBalanceFactor = 0.7
        # waitThresholdFactor = 4
        # [2.9 3.0 0.1 0.8 2 5 0.6 4]
        timeFactor = 1.5
        timeInfluFactor = 0.6
        numSRTInfluence = 0.3
        mrtWeightBalanceFactor = 0.7
        waitThresholdFactor = 6
        mrtDistanceFactor = 3
        waitFactor = 0.6
        acceptFactor = 3

        encouragementFactor = 1.0
        numRobotsSpeculated = 6
        numKeepMRT = 0.1

        numShipPort = 3
        weightRangeSrt = [0, 50]
        randomSeed = 5
        emergencyProportion = [0, 0]

        iteration = 1

        emptyModel = Model(numRobots, limitX, limitY, numTruck, numSrt, numMrt, agentPayload, avgVelocity,
                           numRobotsNeeded,
                           timeFactor, timeInfluFactor, encouragementFactor, numSRTInfluence,
                           mrtWeightBalanceFactor, waitThresholdFactor, mrtDistanceFactor, waitFactor, acceptFactor,
                           numRobotsSpeculated, numKeepMRT,
                           numShipPort, weightRangeSrt, requiredRobots, randomSeed, emergencyProportion)

        for j in range(60000):
            emptyModel.step()
            if j % 1000 == 0:
                numAllocatedSRTList.append(emptyModel.numAllocatedSRT)
                numAllocatedMRTList.append(emptyModel.numAllocatedMRT)
        for j in range(numShipPort):
            print("This is the real time truck schedule of ship port " + str(j))
            print(emptyModel.realTimeSchedule[j])
            print("This is the planned truck schedule of ship port " + str(j))
            print(emptyModel.truckScheduleList[j])

        print("This is the truck delay rate: " + str(emptyModel.truckDelayRate * 100))
        print("This is the average of truck delay time: " + str(emptyModel.truckDelayTime / (3 * numTruck)))
        print("This is the srt delay rate: " + str(emptyModel.srtDelayRate * 100))
        print("This is the mrt delay rate: " + str(emptyModel.mrtDelayRate * 100))
        print("This is the task delay rate: " + str(emptyModel.taskDelayRate * 100))
        print("The traveling distance of robot is: " + str(emptyModel.travelDistanceList))
        print("The idle time of robots is: " + str(emptyModel.idleTimeList))
        print("The cumulative computation time is: " + str(sum(emptyModel.computationTime) / numRobots))
        print("The total waiting time is: " + str(emptyModel.totalWaitTime))
        print("The length of record time is:" + str(len(emptyModel.recordTime)))

        fig, ax = plt.subplots()
        # make a plot
        xPoint = list(range(15))
        ax.plot(xPoint, numAllocatedSRTList[:15], color="red", linestyle='solid', label="SRT", linewidth=2)
        ax.plot(xPoint, numAllocatedMRTList[:15], color="blue", linestyle='solid', label="MRT", linewidth=2)
        ax.set_xlabel("time (1000)", fontsize=14)
        ax.set_ylabel("Number of allocated tasks", fontsize=14)
        ax.legend()
        plt.title('num_SRT=40, numMRT=40')
        plt.show()
        # fig.savefig(
        #     'truckDelayTimeList_40srt_40mrt.jpg',
        #     format='jpeg', dpi=600, bbox_inches='tight')
        # plt.figure(1)
        # plt.title("Robots = 10, task = 150")
        # plt.plot(emptyModel.recordTime, emptyModel.computationTimeRobot)
        # # plt.plot(xpoints, smallRobotsTravelDistanceList, label="Hetero(small)")
        # # plt.plot(xpoints, largeRobotsTravelDistanceList, label="Hetero(large)")
        # plt.ylabel('Computing Time')
        # plt.xlabel('Mission Time (sec)')
        # plt.show()
