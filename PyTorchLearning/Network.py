##import torch
##from torch.autograd import Variable
import json
from pprint import pprint

dataset = []

with open('CreditScoreData.json') as f:
    dataset = json.load(f)

pprint(dataset)


'''
learning_rate = 0.01

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = Variable(torch.Tensor([3.0]), requires_grad = True)## any random values inside the tensor as of rn

for generation in range(25):
    for x_Val, y_Val in zip(x_data, y_data):
        l = loss(x_Val, y_Val)
        l.backward()
        print("\t grad: ", x_Val, y_Val, w.grad.data[0])
        w.data =  w.data - learning_rate * w.grad.data
        w.grad.data.zero_()
    print("Progress=> Generation", generation, "W =", l.data[0])

print(w)
'''