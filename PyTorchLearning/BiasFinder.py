import TwinDataGenerator
import torch

model = torch.load("Model.pt")
experimenter = TwinDataGenerator("CreditScoreData.json")
print(model.forward(torch.Tensor([1, 1, 1, 1, 1, 1])))


