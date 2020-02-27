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
import opt.example.CountOnesEvaluationFunction as CountOnesEvaluationFunction
from array import array




"""
Commandline parameter(s):
   none
"""

N = 100
fill = [2] * N
ranges = array('i', fill)


"""
Q: I am trying to understand what the count ones optimization function is doing.
From the test class in ABAGAIL, I see that an array is created and filled up with all int=2.
I think i understand the concept of counting all of the 1s in the vector,
but I do not see how the array of all 2s turns into an array of 1s and 0s?

If you dig through the classes that uses the ranges variable, it will become clear.
The 2 specifies how many different values are possible at any point in the vector (0 and 1).
"""

ef = CountOnesEvaluationFunction()
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)





rhc = RandomizedHillClimbing(hcp)
print "Random Hill Climb"
iters_list = [10, 25, 50, 100, 200, 300, 400, 500]
for iters in iters_list:
   fit = FixedIterationTrainer(rhc, iters)
   start = time.time()
   fit.train()
   duration = time.time() - start
   print "Iterations: " + str(iters) + ", Fitness: " + str(ef.value(rhc.getOptimal())), ", Duration: " + str(duration)

print "Simulated Annealing"
for iters in iters_list:
   # Initial temperature of 100, with a cooling factor of .95 every iteration.
   # Thus the temperature will be 100 for the first iteration, 95 for the second, 90.25, 87.65...
   sa = SimulatedAnnealing(100, .95, hcp)
   fit = FixedIterationTrainer(sa, iters)
   start = time.time()
   fit.train()
   duration = time.time() - start
   print "Iterations: " + str(iters) + ", Fitness: " + str(ef.value(sa.getOptimal())), ", Duration: " + str(duration)

print "Simulated Annealing - Temperatures"
temps = [100, 90, 75, 50, 25, 10]
# for temp in temps:
# temp = 25
cooling_rate = 0.95
for temp in temps:
   for iters in iters_list:
      sa = SimulatedAnnealing(temp, cooling_rate, hcp)
      fit = FixedIterationTrainer(sa, iters)
      start = time.time()
      fit.train()
      duration = time.time() - start
      print "Iters: " + str(iters) + ", Temp: " + str(temp) + ", Fitness: " + str(ef.value(sa.getOptimal())), ", Duration: " + str(duration)

print "Standard Genetic Algorithm"
for iters in iters_list:
   # Population of size 20, of which 20 will mate and 0 will mutate.
   ga = StandardGeneticAlgorithm(200, 200, 0, gap)
   fit = FixedIterationTrainer(ga, 300)
   fit.train()
   print "Iters: " + str(iters) + ", Fitness " + str(ef.value(ga.getOptimal()))

print "MIMIC Algorithm"
mimic = MIMIC(50, 10, pop)
fit = FixedIterationTrainer(mimic, 100)
fit.train()
print "MIMIC: " + str(ef.value(mimic.getOptimal()))
