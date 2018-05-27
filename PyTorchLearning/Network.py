import torch
from torch.autograd import Variable
import json
import numpy as np
from pprint import pprint


############ Helper methods ####################

def generateDictFromDiscrete(optionsArr):
    result = {}
    denom = 1.0 * len(optionsArr) - 1
    ctr = 0
    for key in optionsArr:
        result[key] = ctr/denom
        ctr += 1
    return result

def rows(numpyArr):
    np.shape(numpyArr)[0]
def cols(numpyArr):
    np.shape(numpyArr)[1]

######### Sigmoid/Relu functions for The different vairables############

def sigmoid_race(race_str):
    return racesDict[race_str]

def sigmoid_education(educ_str):
    return educationDict[educ_str]

def sigmoid_citizenship(citizenship_str):
    if citizenship_str == "b":
        return 2
    else:
        return citizenshipDict[citizenship_str]

def sigmoid_age(age_float):
    if age_float <= 18:
        return 0
    elif age_float >= 70:
        return 1
    else:
        return (1.0/52)*(age_float - 18)

def sigmoid_name(name_string):
    length = len(name_string)
    if length <= 4:
        return 0
    elif length >= 24:
        return 1
    else:
        return (1.0/20)*(length - 4)
def sigmoid_gender(gender_int):
    return gender_int

######### statics ##################
citizenships = ['ar_EG', 'ar_PS', 'ar_SA', 'bg_BG', 'cs_CZ', 'de_DE', 'dk_DK', 'el_GR', 'en_AU', 'en_CA', 'en_GB', 'en_US', 'es_ES', 'es_MX', 'et_EE', 'fa_IR', 'fi_FI', 'fr_FR', 'hi_IN', 'hr_HR', 'hu_HU', 'it_IT', 'ja_JP', 'ko_KR', 'lt_LT', 'lv_LV', 'ne_NP', 'nl_NL', 'no_NO', 'pl_PL', 'pt_BR', 'pt_PT', 'ro_RO', 'ru_RU', 'sl_SI', 'sv_SE', 'tr_TR', 'uk_UA', 'zh_CN', 'zh_TW', 'ka_GE']
citizenshipDict = generateDictFromDiscrete(citizenships)
races = ['almond', 'oreo', 'quaker']
racesDict = generateDictFromDiscrete(races)
educ = ['High School', 'college', 'masters', 'phd', 'genius']
educationDict = generateDictFromDiscrete(educ)

####### Network Object Definition ##############

class Network(torch.nn.Module):
    activation_func = torch.nn.Sigmoid

    def __init__(self):

        super(Network, self).__init__()

        self.layer1 = torch.nn.Linear(6, 12)
        self.layer2 = torch.nn.Linear(12, 12)
        self.layer3 = torch.nn.Linear(12, 12)
        self.layer4 = torch.nn.Linear(12, 10)
        self.layer5 = torch.nn.Linear(10, 5)
        self.layer6 = torch.nn.Linear(5, 4)
        self.layer7 = torch.nn.Linear(4, 3)

    def forward(self, in_data):

        out1 = self.activation_func(self.layer1(in_data))
        out2 = self.activation_func(self.layer2(out1))
        out3 = self.activation_func(self.layer3(out2))
        out4 = self.activation_func(self.layer4(out3))
        out5 = self.activation_func(self.layer5(out4))
        out6 = self.activation_func(self.layer6(out5))
        out7 = self.activation_func(self.layer7(out6))

        prediction = out7
        return prediction



### Main ###

dataset = []

with open('CreditScoreData.json') as f:
    dataset = json.load(f)

stats = dataset[-1]
dataset = dataset[:-1]

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

training_data_in = np.zeros((len(dataset), 6))
training_data_out = np.zeros((len(dataset), 3))


for i in range(len(dataset)):
    training_data_in[i][0]= sigmoid_name(dataset[i]['name'])
    training_data_in[i][1] = sigmoid_citizenship(dataset[i]['citizenship'])
    training_data_in[i][2] = sigmoid_gender(dataset[i]['gender'])
    training_data_in[i][3] = sigmoid_education(dataset[i]['education'])
    training_data_in[i][4] = sigmoid_age(dataset[i]['age'])
    training_data_in[i][5] = sigmoid_race(dataset[i]['race'])

    a = dataset[i]['credit_score']
    if (a < stats['quartile1']):
        training_data_out[i][0] = 1
    elif(a > stats['quartile3']):
        training_data_out[i][2] = 1
    else:
        training_data_out[i][1] = 1


print(training_data_in)
print(training_data_out)

in_data = Variable(torch.from_numpy(training_data_in)).float()
out_data = Variable(torch.from_numpy(training_data_out)).float()


network = Network()
network.float()

criterion_func = torch.nn.BCELoss(size_average=True) ### Try and change this to BCE Loss

print("parameters: ",  network.parameters())

optimizer = torch.optim.SGD(network.parameters(), lr = 0.15)

for epoch in range(len(dataset)):
    print(epoch)

    out_pred = network.forward(in_data[epoch])

    loss = criterion_func(out_pred, out_data[epoch])
    print(epoch, loss.data[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(epoch)

print(network.forward(in_data[0]))

