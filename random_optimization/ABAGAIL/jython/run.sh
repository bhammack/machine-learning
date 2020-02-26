#!/bin/bash
# edit the classpath to to the location of your ABAGAIL jar file
#
export CLASSPATH=../ABAGAIL.jar:$CLASSPATH
mkdir -p data/plot logs image

echo "count ones"
jython countones.py

# echo "four peaks"
# jython fourpeaks.py

# echo "Running traveling salesman test"
# jython travelingsalesman.py

# continuous peaks
# echo "continuous peaks"
# jython continuouspeaks.py

# knapsack
# echo "Running knapsack"
# jython knapsack.py

# abalone test
# echo "Running abalone test"
# jython abalone_test.py
