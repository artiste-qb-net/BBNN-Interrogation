import torch
from pprint import pprint
import json
import random
import numpy as np
from torch.autograd import Variable
from Network import Network
from TwinDataGenerator import TwinDataGenerator


class BiasFinder():

    activation_func = torch.nn.Sigmoid()

    def set_activation_func(torch_function):
        activation_func = torch_function

    def __init__(self, model_file, json_file):
        self.model = Network()
        self.model.load_state_dict(torch.load(model_file))
        self.experiment_gen = TwinDataGenerator(json_file)
        self.output = self.gen_all_cond_probs()



    def approx_forward(self, input_torch_tensor):
        out_arr = self.model.forward(input_torch_tensor)
        largest = max(out_arr)
        for i in range(len(out_arr)):
            if out_arr[i] == largest:
                out_arr[i] = 1
            else:
                out_arr[i] = 0
        return out_arr

    def gen_attr_value_cond_prob(self, attr_index, attr_value):
        result = {}
        arr = self.experiment_gen.gen_data_with_attr(attr_index, attr_value)
        output_size = len(self.model.forward(arr[0]))
        print(output_size)
        a = []
        for i in range(output_size):
            a.append(0)

        for i in arr:
            res = self.approx_forward(i)
            indexofone = 0
            for j in range(len(res)):
                if res[j] == 1:
                    indexofone = j
                    break
            a[indexofone] += 1


        a = [1.0/len(arr) * i for i in a]
        print(a)

        for i in range(len(a)):
            key = "P(Output index is " + str(i) + "| Input index " + str(attr_index) + " has value " + str(attr_value) +")"
            result[key] = a[i]
        return result

    def gen_attr_cond_probs(self, attr_index):
        result = {}
        for i in self.experiment_gen.cardinalities[attr_index]:
            result.update(self.gen_attr_value_cond_prob(attr_index, i))
        return result

    def gen_all_cond_probs(self):
        result = {}
        for i in range(len(self.experiment_gen.cardinalities)):
            a = self.gen_attr_cond_probs(i)
            result.update(a)

        json_arr = [result]
        with open('ConditionalProbabilities.json', 'w') as outfile:
            json.dump(json_arr, outfile)

        return result



bf = BiasFinder("Model.pt", "CreditScoreData.json")


pprint(bf.gen_all_cond_probs())




