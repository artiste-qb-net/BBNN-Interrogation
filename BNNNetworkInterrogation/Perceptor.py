import numpy as np
import math

class Perceptor:

    nodes = 0
    numNodesNextPerceptor = 0
    currentLayerNodeArr = None # size [nodes]
    weights2DArr = None # size [numNodesNextPerceptor][nodes]
    currentBiases = None # size[numNodesNextPerceptor]


    def __init__(self, currentNumNodes, nextNodes, inputValueArr , weights = None, biases = None):
        currentLayerNodesArr = self.genCurrentLayerNodes(inputValueArr);
        if (nextNodes != 0):
            if (weights != None):
                weights2DArr = weights;
            else:
                weights2DArr = self.genRandomWeights()
            if biases != None:
                self.currentBiases = biases
            else:
                self.currentBiases = self.genRandomBiases();
        nodes = currentNumNodes
        numNodesNextPerceptor = nextNodes

    def __repr__(self):
        str =  "Activiations" +self.currentLayerNodeArr + "\n" + "Weights" +self.weights2DArr
        return str


    def getActivationValues(self):
        return self.currentLayerNodeArr



    def genCurrentLayerNodes(self, inputValueArr):
        assert inputValueArr == True
        assert len(inputValueArr) == self.nodes
        a = []
        for i in inputValueArr:
            a.append(i)

        return np.array(a)

    def genRandomWeights(self):
        weightArr = []
        for i in range(self.numNodesNextPerceptor):
            a = [np.random(0, 10) for i in range(self.nodes)]
            weightArr.append(a)
        return np.array(weightArr)


    def genRandomBiases(self):
        return np.array([np.random(0, 10) for i in range(self.numNodesNextPerceptor)])

    def sigmoid(x):
        return 1/(1 + math.e**(-x))

    def genNextLayerNodes(self):
        sig = np.vectorize(Perceptor.sigmoid())
        a = sig(np.dot(self.weights2DArr, self.currentLayerNodeArr) - self.currentBiases)
        return a.tolist();


