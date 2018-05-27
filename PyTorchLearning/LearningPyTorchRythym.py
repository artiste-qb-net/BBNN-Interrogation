'''
PyTorch Rythym

1) Design ur model in a class with Variables

2) Construct Loss, Forward, and Optimizer (mainly from PyTorch API)

3)Training Cycle (forward, backward, update)

'''

import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([4]))
