"""
Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
find optimal weights to a neural network that is classifying abalone as having either fewer
or more than 15 rings.

Based on AbaloneTest.java by Hannah Lau
"""
from __future__ import with_statement

import os
import csv
import time
import sys

sys.path.append('../ABAGAIL.jar')


from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem

import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm


INPUT_FILE = os.path.join("..", "src", "opt", "test", "abalone.txt")

TRAIN_FILE = os.path.join(".", "optdigits.tra")
TEST_FILE = os.path.join(".", "optdigits.tes")



# INPUT_LAYER = 7
INPUT_LAYER = 64
HIDDEN_LAYER = 5
OUTPUT_LAYER = 10
# OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 1000


def initialize_instances():
    """Read the optdigits CSV data into a list of instances."""
    instances = []

    # Read in the abalone.txt CSV file
    with open(TRAIN_FILE, "r") as f:
        reader = csv.reader(f)

        for row in reader:
            # Creates an instance using a float of every value in the row except the label

            # instance = Instance([float(value) for value in row[:-1]])
            # instance.setLabel(Instance(0 if float(row[-1]) < 15 else 1))

            instance = Instance([float(value) for value in row[:-1]])
            digitclass = int(row[-1])
            classes = [0] * 10 # 10 classes
            classes[digitclass] = 1.0
            # Set the ith index to 1 for whatever the class is...
            instance.setLabel(Instance(classes))



            instances.append(instance)

    # print instances
    print instances[5]
    return instances


def train(oa, network, oaName, instances, measure):
    """Train a given network on a set of instances.

    :param OptimizationAlgorithm oa:
    :param BackPropagationNetwork network:
    :param str oaName:
    :param list[Instance] instances:
    :param AbstractErrorMeasure measure:
    """
    print "\nError results for %s\n---------------------------" % (oaName,)

    for iteration in xrange(TRAINING_ITERATIONS):
        oa.train()

        error = 0.00
        for instance in instances:
            network.setInputValues(instance.getData())
            network.run()

            output = instance.getLabel()
            example = Instance(network.getOutputValues())
            example.setLabel(Instance(network.getOutputValues()))
            # output_values = network.getOutputValues() # this should be a list of activation values...

            # example = Instance(network.getOutputValues())
            # example = Instance(output_values, Instance(output_values.get(0)))
            # example = Instance(output_values, output)
            # print "output_values", output_values
            # print "label/output", output
            # print "weight", example.getWeight()
            # sumosquares = 
            # print "sumosquares", sumosquares
            error += measure.value(output, example)

        print "%0.03f" % error


def main():
    """Run algorithms on the abalone dataset."""
    instances = initialize_instances()
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(instances)

    networks = []  # BackPropagationNetwork
    nnop = []  # NeuralNetworkOptimizationProblem
    oa = []  # OptimizationAlgorithm
    oa_names = ["RHC", "SA", "GA"]
    oa_names = ["RHC", "SA"]
    results = ""

    for name in oa_names:
        classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, HIDDEN_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
        networks.append(classification_network)
        nnop.append(NeuralNetworkOptimizationProblem(data_set, classification_network, measure))

    oa.append(RandomizedHillClimbing(nnop[0]))
    oa.append(SimulatedAnnealing(1E11, .95, nnop[1]))
    # oa.append(StandardGeneticAlgorithm(200, 100, 10, nnop[2]))

    for i, name in enumerate(oa_names):
        start = time.time()
        correct = 0
        incorrect = 0

        train(oa[i], networks[i], oa_names[i], instances, measure)
        end = time.time()
        training_time = end - start

        optimal_instance = oa[i].getOptimal()
        networks[i].setWeights(optimal_instance.getData())

        start = time.time()
        for instance in instances:
            networks[i].setInputValues(instance.getData())
            networks[i].run()

            actualindex = instance.getLabel().getData().argMax()
            predictedindex = networks[i].getOutputValues().argMax()

            print "actual", instance.getLabel()
            print "predicted", networks[i].getOutputValues()
            print "actualindex", actualindex
            print "predictedindex", predictedindex

            if actualindex == predictedindex:
                correct += 1
            else:
                incorrect += 1

        end = time.time()
        testing_time = end - start

        results += "\nResults for %s: \nCorrectly classified %d instances." % (name, correct)
        results += "\nIncorrectly classified %d instances.\nPercent correctly classified: %0.03f%%" % (incorrect, float(correct)/(correct+incorrect)*100.0)
        results += "\nTraining time: %0.03f seconds" % (training_time,)
        results += "\nTesting time: %0.03f seconds\n" % (testing_time,)

    print results


if __name__ == "__main__":
    main()

