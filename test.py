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
        numTruck = 3
        numSrt = [20, 20]
        numMrt = [40, 40]
        agentPayload = 50
        avgVelocity = 1
        numRobotsNeeded = 3

        # [1 0.6 0.7 3 0.8 4 1 1.2 1]
        # [1.5 0.4 0.6 3.5 0.6 3.7 1.  1.4]
        # [2.4 0.7 0.6 3.5 0.7 4.5 1.4 1.2] reallocation
        # [2.  0.5 0.5 3.2 0.6 5.  1.  1.2]
        # [2.14 0.6  0.65 3.59 0.61 4.38 1.37 1.37 0.1 ]
        # [2.5  0.6  0.51 3.44 0.64 3.74 0.97 1.34 0.06]
        # [2.09 0.64 0.89 3.27 0.67 4.92 1.4  1.35 0.15]    numSpeculatedRobots = 6, dynamic,
        # [2.35 0.56 0.73 3.57 0.67 4.87 1.16 1.16 0.05]    numSpeculatedRobots = 10, dynamic,
        # [2.47 0.54 0.63 3.21 0.6  5.   1.46 1.35 0.11]    numSpecRobots = 6, static
        # [2.45 0.69 0.54 3.44 0.61 4.79 0.85 1.38 0.12]    GA-SRT_dynamic-3
        # [2.49 0.64 0.64 2.72 0.61 4.16 1.   1.4  0.1 ]    dynamic-8
        # [1.1 0.6 0.6 3.1 0.7 3.3 1.4 0.9]
        timeFactor = 1.1
        waitFactor = 0.6
        stimulusFactor = 0.6
        mrtDistanceFactor = 3.1
        mrtWeightBalanceFactor = 0.7
        waitThresholdFactor = 3.3
        # srtThresholdFactor = 1.4
        maxStimulusFactor = 1.4
        # differenceFactor = 0.15
        bundleParameter = 1

        encouragementFactor = 1
        numRobotsPredict = 6

        numShipPort = 3
        weightRangeSrt = [0, 50]
        randomSeed = 3,
        emergencyProportion = [0.2, 0.4]

        iteration = 1


        emptyModel = Model(numRobots, limitX, limitY, numTruck, numSrt, numMrt, agentPayload, avgVelocity,
                           numRobotsNeeded,
                           timeFactor, waitFactor, stimulusFactor,
                           mrtDistanceFactor, mrtWeightBalanceFactor, waitThresholdFactor,
                           maxStimulusFactor, encouragementFactor, bundleParameter, numRobotsPredict,
                           numShipPort, weightRangeSrt, requiredRobots, randomSeed, emergencyProportion)


        for j in range(60000):
            emptyModel.step()
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
        print("The cumulative computation time is: " + str(sum(emptyModel.computationTime)/numRobots))
        print("The total waiting time is: " + str(emptyModel.totalWaitTime))
        print("The length of record time is:" + str(len(emptyModel.recordTime)))


        plt.figure(1)
        plt.title("Robots = 10, task = 150")
        plt.plot(emptyModel.recordTime, emptyModel.computationTimeRobot)
        # plt.plot(xpoints, smallRobotsTravelDistanceList, label="Hetero(small)")
        # plt.plot(xpoints, largeRobotsTravelDistanceList, label="Hetero(large)")
        plt.ylabel('Computing Time')
        plt.xlabel('Mission Time (sec)')
        plt.show()






