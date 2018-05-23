import numpy as np
import matplotlib.pyplot as plot
import torch
from torch.autograd import variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = 1

def forward(x, b = 0):
    return w*x+b

def loss(x, y):
    yPred = forward(x)
    return (yPred - y)**2;

wList = []
MSElist = []

'''
for w in np.arange(0, 4.1, 0.1):
    print("w = ", w)
    lsum = 0
    for xVal, yVal in zip(x_data, y_data):
        a = loss(xVal, yVal)
        lsum += a
        print("\t", xVal, yVal, a)
    wList.append(w  )
    MSElist.append(lsum/len(x_data))
    print("MSE =", lsum/len(x_data), "\n")

print(MSElist)
plot.xlabel("W value")
plot.ylabel("MSE value")
plot.plot(wList, MSElist)
plot.show()
'''

learning_rate = 0.1

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = variable(torch.Tensor([1.0]), requires_grad = True)## any random values inside the tensor as of rn

for generation in range(25):
    for x_Val, y_Val in zip(x_data, y_data):
        l = loss(x_Val, y_Val)
        l.backward()
        w.data =  w.data - learning_rate*w.grad.data
        w.grad.data.zero()
    print("Progress=> Generation", generation, "W =", l.data[0])







