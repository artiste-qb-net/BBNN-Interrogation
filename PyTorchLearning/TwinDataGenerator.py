import random
import json
from pprint import pprint
import numpy as np
import torch
from torch.autograd import Variable


class TwinDataGenerator:

    def __init__(self, cardinalities_dict):
        self.options_dict = cardinalities_dict
        self.NUM_SAMPLES = 30 #max(35, 2 * max([len(self.options_dict[i]) for i in self.options_dict]))

    def get_all_options(self):
        options_copy = dict(self.options_dict)
        all_options = TwinDataGenerator.get_all_dict_options(options_copy)
        return all_options

    def get_all_dict_options(dic):
        if len(dic) == 0:
            return
        elif len(dic) == 1:
            key = random.choice(list(dic.keys()))
            value_arr = TwinDataGenerator.removekey(dic, key)
            return [{key : val} for val in value_arr]
        else:
            key = random.choice(list(dic.keys()))
            value_arr = TwinDataGenerator.removekey(dic, key)
            options_so_far = TwinDataGenerator.get_all_dict_options(dic)
            result = []
            for val in value_arr:
                for option in options_so_far:
                    copy = dict(option)
                    copy[key] = val
                    result.append(copy)
            return result

    def removekey(dic, key):
        val = dic[key]
        del dic[key]
        return val

    def option_dict_to_tensor(self, option_dict):
        assert len(option_dict.keys()) == len(self.options_dict.keys())
        result = np.zeros(len(option_dict))
        for i in range(len(option_dict)):
            result[i] = option_dict[i]
        return torch.from_numpy(result)

    def cardinalities(self):
        return dict(self.options_dict)

    def gen_data_with_attr(self, constraints_dict):
        for i in constraints_dict:
            assert i is int and constraints_dict[i] is float
            assert constraints_dict[i] in self.options_dict[i]

        category_data = []

        for i in range(self.NUM_SAMPLES):
            tensor = np.zeros(len(self.options_dict))
            for key in constraints_dict:
                tensor[key] = constraints_dict[key]



            for i in range(tensor.size):
                if tensor[i] == 0:
                    tensor[i] = random.choice(self.options_dict[i])
                else:
                    pass

            category_data.append(torch.from_numpy(tensor))

        return category_data

    def get_tensor_from_dict(dic):
        size = len(dic)
        for i in range(size):
            assert i in dic.keys()

        tensor = np.zeros(size).astype(float)
        for i in dic:
            tensor[i] = dic[i]

        result =  Variable(torch.from_numpy(tensor)).float()
        #result = result.type(torch.DoubleTensor)
        return result







'''
class TwinDataGenerator:
    num_samples = 5000

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

    def gen_data_with_attributes(self, attr_index_value_dict):
        category_data = []
        for i in self.dataset:
            copy = torch.Tensor(i.numpy())
            for i in attr_index_value_dict:
                copy[i] = attr_index_value_dict[i]
            category_data.append(copy)
        return category_data


    def get_dataset(self):
        return self.dataset
'''

### main ###
#generator = TwinDataGenerator({0: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]})
#pprint(generator.get_all_options())

