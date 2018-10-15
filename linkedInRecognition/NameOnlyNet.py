import torch
import unicodedata
from torch.autograd import Variable
from DataInterpreter import DataInterpreter

class NameNet(torch.nn.Module):

    ######## Learning Statics################
    activation_func = torch.nn.Sigmoid()
    criterion_func = torch.nn.BCELoss(size_average=True)  ### Try different loss functions
    learning_rate = 0.17


    def __init__(self, initial_input_size):
        super(NameNet, self).__init__()
        torch.set_default_tensor_type('torch.DoubleTensor')
        self.layer1 = torch.nn.Linear(initial_input_size, 2)
        self.layer2 = torch.nn.Linear(2, 2)
        self.layer3 = torch.nn.Linear(2, 2)
        self.layer4 = torch.nn.Linear(2, 2)

        lr = NameNet.learning_rate
        self.optimizer = torch.optim.SGD(self.parameters(), lr)

    def forward(self, input_data):
        activate = NameNet.activation_func

        input = Variable(input_data, requires_grad = True)

        out1 = activate(self.layer1(input))
        out2 = activate(self.layer2(out1))
        out3 = activate(self.layer3(out2))
        out4 = activate(self.layer4(out3))

        result = out4
        return result

    def back_prop(self, input_tensor, out_actual):

        out_pred = self.forward(input_tensor)

        loss = NameNet.criterion_func(out_pred, out_actual)
        try:
            print(loss.item(), "  \n")
        except RuntimeError:
            pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_once(self, name_str, gender_char):
        name_tensor = self.get_name_tensor(name_str)
        gender_tensor = self.get_gender_tensor(gender_char)
        self.back_prop(name_tensor, gender_tensor)



    def name_forward(self, name_str):
        name_tensor = self.get_name_tensor(name_str)
        result = self.forward(name_tensor)

        if result[0] >= result[1]:
            return "Male"
        else:
            return "Female"



    def get_gender_tensor(self, gender_char):
        gender = gender_char
        if len(gender) > 1:
            gender = gender[0]

        if gender == 'M' or gender == 'm':
            return torch.Tensor([1, 0]).double()
        elif gender == 'F' or gender == 'f':
            return torch.Tensor([0, 1]).double()
        else:
            raise NameError("gender_char must be 'm' or 'f'")


    def get_name_tensor(self, name_str):
        name = unicodedata.normalize('NFKD', name_str).encode('ascii', 'ignore')
        name = name.decode('utf-8')
        name = name.lower()
        print(name)
        if " " in name:
            name = name[0 : name.index(" ")]
        if len(name) < 3:
            name = name[0] + name + name[-1]
        first_letter = DataInterpreter.hash_char(name[0])
        first_two = DataInterpreter.hash_char(name[1])
        first_three = DataInterpreter.hash_char(name[2])
        last_three = DataInterpreter.hash_char(name[-3])
        last_two = DataInterpreter.hash_char(name[-2])
        last_letter = DataInterpreter.hash_char(name[-1])

        return torch.Tensor([first_letter, first_two, first_three, last_three, last_two, last_letter]).double()


    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)
        print("SAVED in ", file_path, "!!! \n")

    def load_model(first_layer_size, file_path):
        new_net = NameNet(first_layer_size)
        new_net.load_state_dict(torch.load(file_path))
        return new_net
