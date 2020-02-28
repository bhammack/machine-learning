# traveling salesman algorithm implementation in jython
# This also prints the index of the points of the shortest route.
# To make a plot of the route, write the points at these indexes
# to a file and plot them in your favorite tool.
import sys
import os
import time

sys.path.append('../ABAGAIL.jar')

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.TravelingSalesmanEvaluationFunction as TravelingSalesmanEvaluationFunction
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
import shared.Instance as Instance
import util.ABAGAILArrays as ABAGAILArrays

from array import array




"""
Commandline parameter(s):
    none
"""

# set N value.  This is the number of points
N = 50
random = Random()

points = [[0 for x in xrange(2)] for x in xrange(N)]
for i in range(0, len(points)):
    points[i][0] = random.nextDouble()
    points[i][1] = random.nextDouble()

ef = TravelingSalesmanRouteEvaluationFunction(points)
odd = DiscretePermutationDistribution(N)
nf = SwapNeighbor()
mf = SwapMutation()
cf = TravelingSalesmanCrossOver(ef)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

iters_list = [100, 500, 1000, 2500, 5000, 7500, 10000, 20000]

print "Random Hill Climb"
rhc = RandomizedHillClimbing(hcp)
for iters in iters_list:
    fit = FixedIterationTrainer(rhc, iters)
    start = time.time()
    fit.train()
    dur = time.time() - start
    print "Iters: " + str(iters) + ", Fitness: " + str(ef.value(rhc.getOptimal())) + ", Dur: " + str(dur)
# print "Route:"
# path = []
# for x in range(0,N):
#     path.append(rhc.getOptimal().getDiscrete(x))
# print path



print "Simulated Annealing"
# 1e13, 0.8, 1e12 0.85, ... dang
temp = 1E13
cooling_rate = 0.90
sa = SimulatedAnnealing(temp, cooling_rate, hcp)
for iters in iters_list:
    fit = FixedIterationTrainer(sa, iters)
    start = time.time()
    fit.train()
    dur = time.time() - start
    print "Iters: " + str(iters) + ", Fitness: " + str(ef.value(sa.getOptimal())) + ", Dur: " + str(dur)
# print "Route:"
# path = []
# for x in range(0,N):
#     path.append(sa.getOptimal().getDiscrete(x))
# print path



# print "Genetic Algorithm"
# # 2000, 1500, 250 gives good results
# # 200, 150, 25
# ga = StandardGeneticAlgorithm(4000, 2500, 550, gap)
# for iters in iters_list:
#     fit = FixedIterationTrainer(ga, iters)
#     start = time.time()
#     fit.train()
#     dur = time.time() - start
#     print "Iters: " + str(iters) + ", Fitness: " + str(ef.value(ga.getOptimal())) + ", Dur: " + str(dur)


# print "Route:"
# path = []
# for x in range(0,N):
#     path.append(ga.getOptimal().getDiscrete(x))
# print path



# for mimic we use a sort encoding
ef = TravelingSalesmanSortEvaluationFunction(points)
fill = [N] * N
ranges = array('i', fill)
odd = DiscreteUniformDistribution(ranges)
df = DiscreteDependencyTree(0.1, ranges)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)



print "MIMIC Algorithm"
# 250, 10? 500, 100?
iters = 500
for samples in [100, 150, 200, 500, 750, 1000]:
    mimic = MIMIC(samples, 10, pop)
    # for iters in iters_list:
    fit = FixedIterationTrainer(mimic, iters)
    start = time.time()
    fit.train()
    dur = time.time() - start
    print "Iters: " + str(iters) + ", Fitness: " + str(ef.value(mimic.getOptimal())) + ", Dur: " + str(dur)


# print "Route:"
# path = []
# optimal = mimic.getOptimal()
# fill = [0] * optimal.size()
# ddata = array('d', fill)
# for i in range(0,len(ddata)):
#     ddata[i] = optimal.getContinuous(i)
# order = ABAGAILArrays.indices(optimal.size())
# ABAGAILArrays.quicksort(ddata, order)
# print order
