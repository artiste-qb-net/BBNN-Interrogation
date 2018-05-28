import torch
from torch.autograd import Variable
import json
import numpy as np
from pprint import pprint
import TwinDataGenerator


####### Network Object Definition ##############

class Network(torch.nn.Module):
    ############ Helper methods ####################
    def generateDictFromDiscrete(optionsArr):
        result = {}
        denom = 1.0 * len(optionsArr) - 1
        ctr = 0
        for key in optionsArr:
            result[key] = ctr / denom
            ctr += 1
        return result

    def rows(numpyArr):
        np.shape(numpyArr)[0]

    def cols(numpyArr):
        np.shape(numpyArr)[1]

    ######### statics ##################
    citizenships = ['ar_EG', 'ar_PS', 'ar_SA', 'bg_BG', 'cs_CZ', 'de_DE', 'dk_DK', 'el_GR', 'en_AU', 'en_CA',
                    'en_GB',
                    'en_US', 'es_ES', 'es_MX', 'et_EE', 'fa_IR', 'fi_FI', 'fr_FR', 'hi_IN', 'hr_HR', 'hu_HU',
                    'it_IT',
                    'ja_JP', 'ko_KR', 'lt_LT', 'lv_LV', 'ne_NP', 'nl_NL', 'no_NO', 'pl_PL', 'pt_BR', 'pt_PT',
                    'ro_RO',
                    'ru_RU', 'sl_SI', 'sv_SE', 'tr_TR', 'uk_UA', 'zh_CN', 'zh_TW', 'ka_GE']
    citizenshipDict = generateDictFromDiscrete(citizenships)
    races = ['almond', 'oreo', 'quaker']
    racesDict = generateDictFromDiscrete(races)
    educ = ['High School', 'college', 'masters', 'phd', 'genius']
    educationDict = generateDictFromDiscrete(educ)

    ######### Sigmoid/Relu functions for The different vairables############

    def sigmoid_race(race_str):
        return Network.racesDict[race_str]

    def sigmoid_education(educ_str):
        return Network.educationDict[educ_str]

    def sigmoid_citizenship(citizenship_str):
        if citizenship_str == "b":
            return 2
        else:
            return Network.citizenshipDict[citizenship_str]

    def sigmoid_age(age_float):
        if age_float <= 18:
            return 0
        elif age_float >= 70:
            return 1
        else:
            return (1.0 / 52) * (age_float - 18)

    def sigmoid_name(name_string):
        length = len(name_string)
        if length <= 4:
            return 0
        elif length >= 24:
            return 1
        else:
            return (1.0 / 20) * (length - 4)

    def sigmoid_gender(gender_int):
        return gender_int

    def genDataSetFromJson(fileName_str):
        dataset = []

        with open(fileName_str) as f:
            dataset = json.load(f)

        stats = dataset[-1]
        dataset = dataset[:-1]

        training_data_in = np.zeros((len(dataset), 6))
        training_data_out = np.zeros((len(dataset), 3))

        for i in range(len(dataset)):
            training_data_in[i][0] = Network.sigmoid_name(dataset[i]['name'])
            training_data_in[i][1] = Network.sigmoid_citizenship(dataset[i]['citizenship'])
            training_data_in[i][2] = Network.sigmoid_gender(dataset[i]['gender'])
            training_data_in[i][3] = Network.sigmoid_education(dataset[i]['education'])
            training_data_in[i][4] = Network.sigmoid_age(dataset[i]['age'])
            training_data_in[i][5] = Network.sigmoid_race(dataset[i]['race'])

            a = dataset[i]['credit_score']
            if (a < stats['quartile1']):
                training_data_out[i][0] = 1
            elif (a > stats['quartile3']):
                training_data_out[i][2] = 1
            else:
                training_data_out[i][1] = 1

        in_data = Variable(torch.from_numpy(training_data_in)).float()
        out_data = Variable(torch.from_numpy(training_data_out)).float()

        return in_data, out_data


    def check(self):
        print(self.forward(self.input_tensor[0]))
        print(self.forward(self.input_tensor[1]))
        print(self.forward(self.input_tensor[2]))
        print("...")
        print(self.forward(self.input_tensor[len(self.input_tensor) - 3]))
        print(self.forward(self.input_tensor[len(self.input_tensor) - 2]))
        print(self.forward(self.input_tensor[len(self.input_tensor) - 1]))

        print(self.input_tensor)
        print(self.output_tensor)
        print("parameters: ", self.parameters())


    activation_func = torch.nn.Sigmoid()
    criterion_func = torch.nn.BCELoss(size_average=True)  ### Try and change this to BCE Loss
    learning_rate = 0.15

    def __init__(self, data_file_json):

        super(Network, self).__init__()

        self.layer1 = torch.nn.Linear(6, 12)
        self.layer2 = torch.nn.Linear(12, 12)
        self.layer3 = torch.nn.Linear(12, 12)
        self.layer4 = torch.nn.Linear(12, 3)


        self.input_tensor, self.output_tensor = Network.genDataSetFromJson(data_file_json)

        self.optimizer = torch.optim.SGD(self.parameters(), self.learning_rate)

        for epoch in range(len(self.input_tensor)):
            out_pred = self.forward(self.input_tensor[epoch])

            loss = Network.criterion_func(out_pred, self.output_tensor[epoch])
            print(epoch, loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        torch.save(self.state_dict(), "Model.pt")

    def forward(self, input_data):

        out1 = self.activation_func(self.layer1(input_data))  # self.layer1(input_data)
        out2 = self.activation_func(self.layer2(out1))  # self.layer2(out1)
        out3 = self.activation_func(self.layer3(out2))  # self.layer3(out2)
        out4 = self.activation_func(self.layer4(out3))  # self.layer4(out3)

        prediction = out4
        return prediction


'''
    data converted to 2D Numpy array
    ----------------------Structure-----------------------
    INPUT DATA TENSOR:
    Index:  0 ||     1     ||  2   ||    3    ||  4  ||  5  || 
          name  citizenship  gender  education   age    race
    1
    -
    2
    -
    3
    -
    4
    .
    .
    .

    OUTPUT DATA TENSOR:

    Index:       0                      1
            < quartile1 || > quartile1 and < quartile3 || > quartile3
    1
    -
    2
    -
    3
    -
    4
    .
    .
    .

'''
### Main ###
net = Network("CreditScoreData.json")



##print("parameters: ",  network.parameters())







