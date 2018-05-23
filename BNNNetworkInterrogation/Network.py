import numpy as np
import Perceptor
class Network:
    perceptorArr = None
    numLayers = 0

    def __init__(self, LayerLengthsArr, initialActiviationsArr):
        self.perceptorArr = []
        self.numLayers = len(LayerLengthsArr)
        first = Perceptor(LayerLengthsArr[0],LayerLengthsArr[1], initialActiviationsArr)
        self.perceptorArr.append(first)
        inputActivations = first.genNextLayerNodes()
        for i in range(len(1, LayerLengthsArr) - 1):
            curr = Perceptor(LayerLengthsArr[i], LayerLengthsArr[i+1], first.genNextLayerNodes)
            self.perceptorArr.append(curr)
            first = curr
        self.perceptorArr.append(Perceptor((LayerLengthsArr[self.numLayers], 0, first.genNextLayerNodes)))


