import scipy.io as sio
import numpy as np
from datetime import datetime, timedelta
from DataInterpreter import DataInterpreter
from FaceRecognizer import FaceRecognizer
import unicodedata
import json
import multiprocessing
import gc
import math

def create_thread(func_name, args_tuple):
    gc.collect()
    return multiprocessing.Process(target = func_name, args = args_tuple)

def convert_mat_date_toint(matlab_date):
    python_datetime = datetime.fromordinal(int(matlab_date)) + timedelta(days=matlab_date % 1) - timedelta(days=366)
    return python_datetime.year


def simplify(str):
    result = unicodedata.normalize('NFKD', str).encode('ascii','ignore')
    return result.lower()



def get_data_file(dataset, thread_num):
    detector = FaceRecognizer()
    temp_dataset = []
    ctr = 0
    file_ctr = 0
    gc.collect()

    while len(dataset) > 0:
        datum = dataset.pop(0)
        print("File ", thread_num, file_ctr, "datapoint", ctr)
        detector.new_image(datum['path'])
        try:
            a = detector.get_face_1D_numpy().tolist()
            datum["as_tensor"] = a
            datum["landmarks"] = detector.get_facial_landmarks(thread_num)
            temp_dataset.append(datum)
            gc.collect()
        except (LookupError, AssertionError, NameError, FileNotFoundError) as e:
            gc.collect()
            pass

        ctr += 1

    file_name = 'FinalDataset/FinalDataset' + str(thread_num) + "t_" + str(file_ctr)+ "f_"+ '.json'
    file_ctr += 1
    with open(file_name, 'w') as outfile:
        json.dump(temp_dataset, outfile)

    del temp_dataset
    gc.collect()
    print("Thread Num: ", thread_num, " Done")


def create_divisions(size, num_divs):
    a = np.linspace(0, size, num_divs)
    print(a)

    for i in range(len(a) - 1):
        a[i] = math.floor(a[i])

    a[-1] = math.ceil(a[-1])
    a = a.astype(int)
    result = []
    for i in range(1, len(a)):
        group = (a[i-1], a[i])
        result.append(group)
    return result

def get_name_tensor(name_str):
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

    return [first_letter, first_two, first_three, last_three, last_two, last_letter]

def main():
    PARENT_PATH = "imdb_crop/"

    raw_data = sio.loadmat("imdb_crop/imdb.mat")
    final_dataset = []
    dataset = raw_data["imdb"][0:100000]## REMEMBER TO CHANGE FOLDER LOCATION WHEN CHANGIN THIS

    matlab_dates = dataset[0][0][0][0]

    size = matlab_dates.size

    year_photo_taken_arr = dataset[0][0][1][0]

    np_ages = np.zeros(size)

    for i in np.arange(np_ages.size):
        try:
            birth_year = convert_mat_date_toint(matlab_dates[i].item())
        except OverflowError:
            birth_year = year_photo_taken_arr[i].item()


        np_ages[i] = year_photo_taken_arr[i].item() - birth_year

    np_genders = dataset[0][0][3][0]
    np_file_paths = np.empty([size], dtype=object)
    relative_path = dataset[0][0][2][0]
    for i in np.arange(np_file_paths.size):
        np_file_paths[i] = PARENT_PATH + relative_path[i].item()

    names = dataset[0][0][4][0]

    np_names = np.empty([size], dtype=object)

    for i in range(np_names.size):
        try:
            np_names[i] = simplify(names[i][0].item()).decode("utf-8")
            np_names[i] = get_name_tensor(str(np_names[i]))
        except LookupError as e:
            np_names[i] = None

    for i in range(np_names.size):
        data_point = {'name': np_names[i], 'age': np_ages[i].item(), 'gender': np_genders[i].item(), 'path': np_file_paths[i]}
        if (np_ages[i] > 0 or np_names[i] == None):
            final_dataset.append(data_point)

    del (raw_data)
    del (dataset)
    del (year_photo_taken_arr)
    del (np_ages)
    del (np_genders)
    del (np_file_paths)
    del (np_names)
    del (relative_path)
    del (names)
    gc.collect()



    thread_ctr = 1
    max_threads = 6
    thread_arr = []

    while len(final_dataset) != 0:
        next_cut = min(len(final_dataset), 1000)

        for i in range(max_threads):
            if (len(final_dataset) <= 0):
                break

            mini_dataset = final_dataset[0: next_cut]
            final_dataset = final_dataset[next_cut: len(final_dataset)]
            t = create_thread(get_data_file, (mini_dataset, thread_ctr))
            thread_ctr += 1
            thread_arr.append(t)

        for th in thread_arr:
            th.start()

        for th in thread_arr:
            th.join()
            th.terminate()
            thread_arr = []






    '''
    pairs = create_divisions(size, 13)
    dataset_divs = []
    for pair in pairs:
        dataset_divs.append(final_dataset[(pair[0] + 1): (pair[1])])

    print(pairs)
    print(len(pairs))
    print(len(dataset_divs))


    del(PARENT_PATH)
    del(raw_data)
    del(dataset)
    del(year_photo_taken_arr)
    del(np_ages)
    del(np_genders)
    del(np_file_paths)
    del(np_names)
    del(relative_path)
    del(names)
    del(pairs)
    gc.collect()


    thread_arr = []
    for i in range(12):
        t = create_thread(get_data_file, (dataset_divs[i], i))
        thread_arr.append(t)
    

    for th in thread_arr:
        th.start()


    for th in thread_arr:
        th.join()

    print("done")
    '''



if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
