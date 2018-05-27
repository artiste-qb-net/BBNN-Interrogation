import Test
import json
from pprint import pprint
import time

class TwinDataGenerator:
    numSamples = 125
    dataset = []
    

    def __init__(self, datasetArr):
        finalData = []
        for i in range(self.numSamples):
            finalData.append(dict(Test.choice(datasetArr)))
        self.dataset = finalData
    

    def genDiscreteTwinData(self, optionsArr, attrName):
        twinData = {}
        for key in optionsArr:
            twinData[str(attrName) + ' is ' + str(key)] = self.genDataWithAttr(attrName, key)
        return twinData


    def genDataWithAttr(self, attrName, attrValue):
        assert len(self.dataset) > 0 and self.dataset[0].__contains__(attrName)
        categoryData = []
        for i in self.dataset:
            copy = {}
            copy = dict(i)
            copy[attrName] = attrValue
            categoryData.append(copy)
            del copy['credit_score']
        return categoryData


### Main ###

dataset = []

with open('CreditScoreData.json') as f:
    dataset = json.load(f)

stats = dataset[-1]
dataset = dataset[:-1]



pprint(stats)

time.sleep(1)

dataset = dataset[:len(dataset) -1]

twinDataGen = TwinDataGenerator(dataset)

twinData = twinDataGen.genDiscreteTwinData(['High School', 'college', 'masters', 'phd', 'genius'], 'education')

pprint(twinData)