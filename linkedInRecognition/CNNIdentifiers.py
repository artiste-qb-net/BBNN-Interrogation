import torch
import torch.nn as nn
import random as r
from FaceRecognizer import FaceRecognizer
import torch.nn.functional as F
import math
import torchvision
import torchvision.transforms as transforms
import json
import numpy as np
from torch.autograd import Variable
from pprint import pprint

class AgeNet(torch.nn.Module):
    ################################# STATIC METHODS  ################################

    def create_out_data_point(datum_dict):
        age = datum_dict['age']
        gender = datum_dict['gender']
        if math.isnan(gender):
            gender = r.randint(0, 1)

        if math.isnan(age) or math.isinf(age):
            age = 20

        if age > 99:
            age = 99

        if age < 5:
            age = 0

        output = [[gender * 100 + age]]

        return torch.Tensor(output)



    '''
    Just me trying to create my own loss function
    
    
    def custom_loss(out_predicted_tensor, desired_out_tensor):
        #assert out_predicted_tensor.shape() == desired_out_tensor.shape()
        out_predicted_tensor = out_predicted_tensor.reshape(out_predicted_tensor.size())
        desired_out_tensor = desired_out_tensor.reshape(desired_out_tensor.size())
        loss = 0

        for i in range(out_predicted_tensor.size()):
            delta = abs(out_predicted_tensor[i] - desired_out_tensor[i])
            delta = 5*math.sqrt(delta / 5.0)
            loss += delta

        return loss

    '''





    ################################# INSTANCE METHODS ###############################
    def __init__(self):
        super(AgeNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size= 5, stride = 3, padding = 2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size= 3, stride= 2, padding = 1)
        #self.conv3 = nn.Conv2d(256, 384, kernel_size= 3, stride= 3, padding= 1)

        self.pool1 = nn.MaxPool2d(kernel_size = 5, stride = 1, padding = 0)
        self.pool2 = nn.MaxPool2d(kernel_size= 3, stride = 1, padding = 0)
        #self.pool3 = nn.MaxPool2d(kernel_size= 2, stride = 2, padding = 0)


        self.fc1 = nn.Linear(21*21*256, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)


        self.activation_func = F.relu#Didnt notice much difference between sigmoid and relu
        self.criterion_func = nn.MSELoss(size_average=True) ### Try different loss functions
        self.learning_rate = 0.025 #have tried around 75 times between 0.025 and 2.5*10^-7

    def forward(self, in_data):

        activate = self.activation_func

        result = Variable(in_data.float())
        #print(result.size())

        result = activate(self.conv1(result))
        #print(result.size())

        result = self.pool1(result)
        #print(result.size())

        result = activate(self.conv2(result))
        #print(result.size())

        result = self.pool2(result)
        #print(result.size())

        result = result.view(1,-1).float()

        result = F.sigmoid(self.fc1(result))
        result = F.sigmoid(self.fc2(result))
        result = self.fc3(result)
        #result = self.fc4(result)

        print("Result: ")
        pprint(result)


        return result

    '''
    This method is the prefered way of using the forward pass. Just specify file path
    '''
    def forward_img(self, img_filepath):
        fr = FaceRecognizer()
        fr.new_image(img_filepath)
        np_face_arr = fr.get_face_as_numpy()
        input_tensor = torch.from_numpy(np_face_arr)
        return self.forward(input_tensor)

    ## add

    def load_input_and_output_tensor(json_file_path):
        raw_data = []


        with open(json_file_path) as json_file:
            raw_data = json.load(json_file)

        data_len = len(raw_data)

        assert len(raw_data) != 0, "No training data in file"

        in_data = []
        out_data = []

        for i in range(data_len):
            print(i)
            datum = raw_data[i]
            in_data_point = np.array(datum["as_tensor"])

            '''
            for i in range(in_data_point.size):
                in_data_point[i] = ((i/255) -0.5) * 8
            '''

            in_data_point = in_data_point.reshape(150, 150, 3)
            in_data_point = in_data_point.swapaxes(0, 2)
            assert in_data_point.shape == (3, 150, 150)
            in_data_point = torch.from_numpy(in_data_point)
            in_data_point.unsqueeze_(0)
            in_data.append(in_data_point)
            out_data_point = AgeNet.create_out_data_point(datum)
            out_data.append(out_data_point)
            print("Age: ", datum['age'], "Gender: ", datum['gender'])

        return in_data, out_data


    def train(self, data_file_path):
        input_tensor, output_tensor = AgeNet.load_input_and_output_tensor(data_file_path)

        epoch = 0
        while len(input_tensor) != 0 and len(output_tensor) != 0:
            assert len(input_tensor) == len(output_tensor)

            rand_point = r.randint(0, len(input_tensor) -1)

            print("epoch: ", epoch)
            out_actual = output_tensor.pop(rand_point)
            curr_input = input_tensor.pop(rand_point)
            self.back_prop(curr_input, out_actual)
            epoch += 1

    def train_once(self, datum_dict):
        in_data_point = torch.Tensor(datum_dict["as_tensor"])
        in_data_point = in_data_point.view(150, 150, 3)
        out_data_point = torch.Tensor([datum_dict['age'], datum_dict['gender']* 10])
        self.back_prop(in_data_point, out_data_point)

    def back_prop(self, input_tensor, out_actual):
        out_pred = self.forward(input_tensor)
        print("Actual: ")
        pprint(out_actual)
        print("\n")

        optimizer = torch.optim.SGD(self.parameters(), lr = self.learning_rate, momentum=0.9)
        #out_actual = out_actual.long()
        #loss = self.criterion_func(out_pred, torch.max(out_actual, 1)[1]) * 100
        loss = self.criterion_func(out_pred, out_actual)

        optimizer.zero_grad()
        loss.backward()

        print("Loss: ", loss.item())
        #print("Grad: ", input_tensor.grad, "\n")
        for param in self.parameters():
            print(param.grad.data.sum())



        print("\n \n")
        optimizer.step()

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)
        print("SAVED in ", file_path, "!!! \n")


    def load_model(file_path):
        new_net = AgeNet()
        new_net.load_state_dict(torch.load(file_path))
        return new_net

######################################################################################################################################


