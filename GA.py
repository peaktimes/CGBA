import pygad
from ProposedModel import Model
import numpy
import csv
import matplotlib


def fitness_func(solution, solution_idx):
    fitness = 0
    for k in range(5):
        numRobots = 10
        requiredRobots = 12
        limitX = [0, 300]
        limitY = [0, 100]
        numTruck = 3
        numSrt = [20, 20]
        numMrt = [50, 50]
        agentPayload = 50
        avgVelocity = 1
        numRobotsNeeded = 3

        timeFactor = solution[0]
        timeInfluFactor = solution[1]
        numSRTInfluence = solution[2]
        mrtWeightBalanceFactor = solution[3]
        waitThresholdFactor = solution[4]
        mrtDistanceFactor = solution[5]
        waitFactor = solution[6]
        acceptFactor = solution[7]

        numRobotsSpeculated = 5
        encouragementFactor = 1
        numKeepMRT = 0.1

        numShipPort = 3
        weightRangeSrt = [0, 50]
        emergencyProportion = [0.2, 0.4]

        randomSeed = k
        emptyModel = Model(numRobots, limitX, limitY, numTruck, numSrt, numMrt, agentPayload, avgVelocity,
                           numRobotsNeeded,
                           timeFactor, timeInfluFactor, encouragementFactor, numSRTInfluence,
                           mrtWeightBalanceFactor, waitThresholdFactor, mrtDistanceFactor, waitFactor, acceptFactor,
                           numRobotsSpeculated, numKeepMRT,
                           numShipPort, weightRangeSrt, requiredRobots, randomSeed, emergencyProportion)

        for i in range(60000):
            emptyModel.step()

        fitness += 100 / (emptyModel.truckDelayTime + 0.1)
    fitness /= 6
    return fitness


num_generations = 35
num_parents_mating = 20
sol_per_pop = 40
num_genes = 8

fitness_function = fitness_func

parent_selection_type = "rws"

crossover_type = "single_point"
crossover_probability = 0.3

mutation_type = "adaptive"
mutation_probability = [0.25, 0.1]

# parallel_processing = ["thread", 20]

# random_mutation_min_val = -0.1
# random_mutation_max_val = 0.1

# [2.4 0.7 0.6 3.5 0.7 4.5 1.4 1.2]
# [2.5  0.6  0.51 3.44 0.64 3.74 0.97 1.34 0.06]


timeFactorRange = {'low': 1, 'high': 4}
timeInfluFactorRange = {'low': 0.5, 'high': 2}
numSRTInfluenceRange = {'low': 0.1, 'high': 0.4}
mrtWeightBalanceFactorRange = {'low': 0.7, 'high': 0.9}
waitThresholdFactorRange = {'low': 2, 'high': 4}
mrtDistanceFactorRange = {'low': 2, 'high': 4}
waitFactorRange = {'low': 0.5, 'high': 0.9}
acceptFactorRange = {'low': 2, 'high': 4}
# timeFactorRange = {'low': 0.7, 'high': 1.5}
# waitFactorRange = {'low': 0.4, 'high': 0.8}
# stimulusFactorRange = {'low': 0.5, 'high': 1.0}
# mrtDistanceFactorRange = {'low': 2, 'high': 4}
# mrtWeightBalanceFactorRange = {'low': 0.6, 'high': 0.9}
# waitThresholdFactorRange = {'low': 2.5, 'high': 5}
# maxStimulusFactorRange = {'low': 1, 'high': 1.4}
# bundleParameterRange = {'low': 0.8, 'high': 1.2}
# numKeepMRTRange = {'low': 0.1, 'high': 0.2}

gene_space = [timeFactorRange, timeInfluFactorRange, numSRTInfluenceRange, mrtWeightBalanceFactorRange,
              waitThresholdFactorRange, mrtDistanceFactorRange, waitFactorRange, acceptFactorRange]
gene_type = [[float, 1], [float, 1], [float, 1], [float, 1], int, int, [float, 1], int]
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_type=gene_type,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       crossover_probability=crossover_probability,
                       mutation_type=mutation_type,
                       mutation_probability=mutation_probability,
                       gene_space=gene_space,
                       keep_elitism=1,
                       allow_duplicate_genes=False
                       )
ga_instance.run()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
ga_instance.plot_fitness()
ga_instance.save("taskAllocation")

# with open(
#         'C:\\Users\\34783\\Desktop\\reference book\\final project\\programming\\taskAllocation\\parameter adjustment\\numSpecRobot_6',
#         'w',
#         encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(solution)
#     writer.writerow(solution_fitness)



# Saving the GA instance.
# filename = 'ga_10'  # The filename to which the instance is saved. The name is without extension.
# ga_instance.save(filename=filename)

# Loading the saved GA instance.
# loaded_ga_instance = pygad.load(filename=filename)
# loaded_ga_instance.plot_fitness()
