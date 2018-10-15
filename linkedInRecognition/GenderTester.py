import torch
from ImageClassifier import GenderNetwork
import json
from pprint import pprint
from DataInterpreter import DataInterpreter


net = GenderNetwork(67506)
net.train("FinalDataset.json")
net.save_model("ModelGender.pt")

net_structure = torch.load("ModelGender.pt")
pprint(net_structure)
print("..........................................")
trained_net = GenderNetwork.load_model(67506, "ModelGender.pt")
raw_data = []
with open("FinalDataset.json") as json_file:
    raw_data = json.load(json_file)

test_point = raw_data[4]

pprint(test_point)
print("\n")
interpreter = DataInterpreter()
interpreter.new_datapoint(test_point)
input = interpreter.get_image_and_name_tensor()

result = trained_net.forward(input)

pprint(result)