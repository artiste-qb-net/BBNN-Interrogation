import random
import json
from pprint import pprint
import time

class TwinDataGenerator:
    numSamples = 1000


    def init_with_dataset(self, datasetArr):
        finalData = []
        for i in range(self.numSamples):
            finalData.append(dict(random.choice(datasetArr)))
        self.dataset = finalData

    def __init__(self, file_name_json):
        self.dataset = None
        with open(file_name_json) as f:
            self.dataset = json.load(f)

        self.stats = self.dataset[-1]
        self.dataset = self.dataset[:-1]
        self.init_with_dataset(self.datasetdataset)



    def genDiscreteTwinData(self, optionsArr, attrName):
        twinData = {}
        for key in optionsArr:
            twinData[str(attrName) + ' is ' + str(key)] = self.genDataWithAttr(attrName, key)
        return twinData


    def genDataWithAttr(self, attrName, attrValue):
        assert len(self.dataset) > 0 and self.dataset[0].__contains__(attrName)
        categoryData = []
        for i in self.dataset:
            copy = dict(i)
            copy[attrName] = attrValue
            categoryData.append(copy)
            del copy['credit_score']
        return categoryData


### Main ###
