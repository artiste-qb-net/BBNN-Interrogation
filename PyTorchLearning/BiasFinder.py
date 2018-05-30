import torch
from pprint import pprint
import json
import random
import numpy as np


class BiasFinder():
    def __init__(self, model_file, json_file):
        model = torch.load(model_file)
        self.weight_arr = []
        self.bias_arr = []

        ctr = 0
        for key in model:
            if ctr % 2 == 0:
                self.weight_arr.append(model[key].numpy().transpose(1, 0))
            else:
                self.bias_arr.append(model[key].numpy().reshape(len(model[key]), 1))
            ctr += 1



    def forward(self, input_torch_tensor):
        curr_tensor = input_torch_tensor.numpy()
        print(len(self.weight_arr) , "\n")
        for i in range(len(self.weight_arr)):
            print(i)
            curr_tensor = np.dot(curr_tensor, self.weight_arr[i])
            pprint(curr_tensor.shape)
            curr_tensor = curr_tensor - self.bias_arr[i]
        return curr_tensor



class twinDataGenerator:
    num_samples = 1000



    def init_with_dataset(self, dataset_arr):
        final_data = []
        numItems = len(dataset_arr[0])
        self.cardinalities = []
        for c in range(numItems):
            self.cardinalities.append(set())


        for i in range(self.num_samples):
            curr = (dict(random.choice(dataset_arr)))
            del curr['credit_score']
            curr = self.first_activations_func(curr)
            for j in range(len(curr)):
                self.cardinalities[j].add(curr[j])
            final_data.append(curr)
        self.dataset = final_data

    def __init__(self, file_name_json, initial_activation_func_from_dict):
        self.first_activations_func = initial_activation_func_from_dict
        self.dataset = None
        with open(file_name_json) as f:
            self.dataset = json.load(f)
        self.stats = self.dataset[-1]
        self.dataset = self.dataset[:-1]
        self.init_with_dataset(self.dataset)



    def gen_discrete_twin_data(self, attr_index, attr_options):
        twin_data = {}
        for key in optionsArr:
            twin_data[str(attr_name) + ' is ' + str(key)] = self.genDataWithAttr(attr_name, key)
        return twin_data


    def gen_data_with_attr(self, attr_index, attr_value):
        assert len(self.dataset) > 0 and self.dataset.shape[1] >= attr_index
        category_data = []
        for i in self.dataset:
            copy = list(i)
            copy[attr_index] = attr_value
            category_data.append(copy)
        return category_data


bf = BiasFinder("Model.pt", "CreditScoreData.json")

print("...")
print("...")
print("...")

pprint(bf.weight_arr)
pprint(bf.bias_arr)
pprint(bf.forward(torch.Tensor([[0, 0, 0, 0, 0, 0]])))

