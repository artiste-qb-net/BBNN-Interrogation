import torch
from ImageClassifier import AgeNetwork
import json
from pprint import pprint
from CNNIdentifiers import AgeNet
from DataInterpreter import DataInterpreter

net = AgeNet()

num_datasets = 460

for i in range(num_datasets):
    file_num = i+1
    file_path = "FinalDataset/FinalDataset" + str(file_num) + "t_0f_.json"
    net.train(file_path)

net.save_model("ModelAge.pt")

net_structure = torch.load("ModelAge.pt")
pprint(net_structure)
print("..........................................")

