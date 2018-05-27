from torch.autograd import Variable
import torch

a = Variable(torch.Tensor([[3.], [4.], [5.]]))
print(a)
print(a.size()[1])
print(torch.nn.Sigmoid(0))