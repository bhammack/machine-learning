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

from array import array



"""
Commandline parameter(s):
   none
"""

N = 200
T = N / 5
fill = [2] * N
ranges = array('i', fill)

ef = FourPeaksEvaluationFunction(T)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

iters_list = [100, 500, 1000, 2500, 5000, 7500, 10000, 20000]

print "Random Hill Climbing"
rhc = RandomizedHillClimbing(hcp)
for iters in iters_list:
   fit = FixedIterationTrainer(rhc, iters)
   start = time.time()
   fit.train()
   dur = time.time() - start
   print "Iters: " + str(iters) + ", Fitness: " + str(ef.value(rhc.getOptimal())) + ", Dur: " + str(dur)

print "Simulated Annealing"
temp = 100000
cooling_rate = 0.85
sa = SimulatedAnnealing(temp, 0.85, hcp)
for iters in iters_list:
   fit = FixedIterationTrainer(sa, iters)
   start = time.time()
   fit.train()
   dur = time.time() - start
   print "Iters: " + str(iters) + ", Fitness: " + str(ef.value(sa.getOptimal())) + ", Dur: " + str(dur)

print "Genetic Algorithm"
ga = StandardGeneticAlgorithm(200, 175, 20, gap)
for iters in iters_list:
   fit = FixedIterationTrainer(ga, iters)
   start = time.time()
   fit.train()
   dur = time.time() - start
   print "Iters: " + str(iters) + ", Fitness: " + str(ef.value(ga.getOptimal())) + ", Dur: " + str(dur)

print "MIMIC"
mimic = MIMIC(250, 20, pop)
for iters in iters_list:
   fit = FixedIterationTrainer(mimic, iters)
   start = time.time()
   fit.train()
   dur = time.time() - start
   print "Iters: " + str(iters) + ", Fitness: " + str(ef.value(mimic.getOptimal())) + ", Dur: " + str(dur)
