import csv
from NameOnlyNet import NameNet
from pprint import pprint

dataset = []

with open('name_gender.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    removed_head = False
    for row in spamreader:
        if not removed_head:
            removed_head = True
        else:
            new_row = [row[0], row[1]]
            dataset.append(new_row)

with open('gender_refine-csv.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    removed_head = False
    for row in spamreader:
        if not removed_head:
            removed_head = True
        else:
            if (row[1] == '0'):
                gender = 'F'
            else:
                gender = 'M'
            new_row = [row[0], gender]
            dataset.append(new_row)

#pprint(dataset)

test_data = dataset[0:10]
pprint(test_data)
neural_net = NameNet(6)

for i in dataset:
    name = i[0]
    gender = i[1]
    neural_net.train_once(name, gender)


for i in range(3):
    print('... \n')

for i in test_data:
    print(i[0] , "predicted to be: ", neural_net.name_forward(i[0]), "Actually: ", i[1])

print("Elif", "predicted to be: ", neural_net.name_forward("Elif"), "Actually: F")
print("Preeti", "predicted to be: ", neural_net.name_forward("Preeti"), "Actually: F")
print("Aryan", "predicted to be: ", neural_net.name_forward("Aryan"), "Actually: M")
print("Aarushi", "predicted to be: ", neural_net.name_forward("Aarushi"), "Actually: F")


neural_net.save_model("GenderFromName.pt")



