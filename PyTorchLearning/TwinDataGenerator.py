import random
import json
from Network import Network
from pprint import pprint
import torch


class TwinDataGenerator:
    num_samples = 1000

    def init_with_dataset(self, dataset_arr):
        final_data = []
        numItems = len(dataset_arr[0])
        self.cardinalities = []
        for c in range(numItems - 1):
            self.cardinalities.append(set())


        for i in range(self.num_samples):
            curr = (dict(random.choice(dataset_arr)))
            del curr['credit_score']
            curr = self.first_activations_func(curr)
            for j in range(len(curr)):
                self.cardinalities[j].add(curr[j].item())
            final_data.append(curr)
        self.dataset = final_data

    def __init__(self, file_name_json):
        self.first_activations_func = Network.to_data_tensor
        dataset = None
        with open(file_name_json) as f:
            dataset = json.load(f)
        self.stats = dataset[-1]
        dataset = dataset[:-1]
        self.init_with_dataset(dataset)
        print("\n ... \n")

    def gen_data_with_attr(self, attr_index, attr_value):
        assert len(self.dataset) > 0
        assert attr_value in self.cardinalities[attr_index]
        category_data = []
        for i in self.dataset:
            copy = torch.Tensor(i.numpy())
            copy[attr_index] = attr_value
            category_data.append(copy)
        return category_data




### main ###

