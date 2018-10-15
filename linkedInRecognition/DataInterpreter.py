import torch
from pprint import pprint
from torch.autograd import Variable
import numpy as np
import json
from FaceRecognizer import FaceRecognizer


class DataInterpreter:
    num_pixels = FaceRecognizer.FINAL_DIM[0] * FaceRecognizer.FINAL_DIM[1] * 3

    def __init__(self, datapoint={}):
        print(FaceRecognizer.FINAL_DIM[0] * FaceRecognizer.FINAL_DIM[1] * 3)
        self.data_dict = datapoint

    def new_datapoint(self, new_data_dict):
        self.data_dict = new_data_dict

    def hash_str(str):
        hash = 0;
        for i in range(len(str)):
            hash = hash * 27 + (ord(str[i])- 109)
        return hash

    def hash_char(charac):
        hash = (ord(charac)- 109)/(15.0)
        return hash

    def get_name_tensor(self):
        name = ""
        try:
            name = self.data_dict['name']
            if name == None or name == "":
                raise NameError
        except NameError:
            return torch.Tensor([0, 0, 0, 0, 0, 0])

        if " " in name:
            name = name[:name.index(" ")]

        first_letter = DataInterpreter.hash_str(name[0])
        first_two = DataInterpreter.hash_str(name[0:2])
        first_three = DataInterpreter.hash_str(name[0:3])
        last_three = DataInterpreter.hash_str(name[-3:])
        last_two = DataInterpreter.hash_str(name[-2:])
        last_letter = DataInterpreter.hash_str(name[-1])

        return torch.Tensor([first_letter, first_two, first_three, last_three, last_two, last_letter]).double()

    def get_image_tensor(self):
        image_path = self.data_dict['path']
        face_finder = FaceRecognizer()
        face_finder.new_image(image_path)
        np_tensor = np.zeros(DataInterpreter.num_pixels)

        try:
            np_tensor = face_finder.get_face_1D_numpy()
        except (AssertionError, FileNotFoundError) as e:
            pass

        return torch.from_numpy(np_tensor).double()

    def get_image_and_name_tensor(self):
        np_tensor = np.zeros(DataInterpreter.num_pixels + 6)

        image_path = self.data_dict['path']
        face_finder = FaceRecognizer()
        face_finder.new_image(image_path)
        np_face_tensor = np.zeros(DataInterpreter.num_pixels)

        try:
            np_face_tensor = face_finder.get_face_1D_numpy()
        except AssertionError:
            pass

        assert np_face_tensor.size < np_tensor.size

        ctr = 0
        for i in range(np_face_tensor.size):
            np_tensor[i] = np_face_tensor[i]
            ctr += 1


        name = ""
        try:
            name = self.data_dict['name']
            if name == None or name == "":
                raise NameError

            if " " in name:
                name = name[:name.index(" ")]

            first_letter = DataInterpreter.hash_str(name[0])
            first_two = DataInterpreter.hash_str(name[1])
            first_three = DataInterpreter.hash_str(name[2])
            last_three = DataInterpreter.hash_str(name[-3])
            last_two = DataInterpreter.hash_str(name[-2])
            last_letter = DataInterpreter.hash_str(name[-1])

            np_tensor[ctr] = first_letter
            np_tensor[ctr + 1] = first_two
            np_tensor[ctr + 2] = first_three
            np_tensor[ctr + 3] = last_three
            np_tensor[ctr + 4] = last_two
            np_tensor[ctr + 5] = last_letter

        except NameError:
            np_tensor[ctr] = 0
            np_tensor[ctr + 1] = 0
            np_tensor[ctr + 2] = 0
            np_tensor[ctr + 3] = 0
            np_tensor[ctr + 4] = 0
            np_tensor[ctr + 5] = 0

        return torch.from_numpy(np_tensor).double()

    def get_age_tensor(self):
        try:
            age = int(self.data_dict['age'])
            age = min(age, 100)
            age = max(age, 1)
            arr = np.zeros(20)
            arr[int((age - 1)/5)] = 1
            return torch.from_numpy(arr)
        except NameError:
            age = 0
        return torch.Tensor([age])

    def get_gender_tensor(self):
        try:
            gender = self.data_dict['gender']
            if (gender != 0 and gender != 1):
                raise NameError
        except NameError:
            gender = 2

        return torch.Tensor([gender])


    def get_output_tensor(self):
        try:
            age = self.data_dict['age']
        except NameError:
            age = 0

        try:
            gender = self.data_dict['gender']
            if (gender != 0 and gender != 1):
                raise NameError
        except NameError:
            gender = 2

        return torch.Tensor([age, gender])
