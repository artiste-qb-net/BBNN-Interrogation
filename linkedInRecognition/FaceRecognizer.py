import face_recognition as f_r
from PIL import Image
import numpy as np
import os
from io import BytesIO
import base64
import math
from pprint import pprint

'''

User API Pipeline:

1) Create an instance of the object by calling the constructor

2) Call new_image(file_path)on the image you want to analyze

3) Use any of the functions below to access the face (in numpy, 1D numpy, PIL, etc) or features of it.

4) If you want to switch the image you are analyzing, call new_image(file_path) again

5) Continue


'''


class FaceRecognizer:

    '''
    Use this attr to set the Final Image of dimensions of the cropped face.
    Must be square
    WARNING: Upsizing can reduce quality dramatically
    '''

    FINAL_DIM = (150, 150)

    '''
    creates the face recognizer
    '''
    def __init__(self):
        self.pic_path = ""
        assert FaceRecognizer.FINAL_DIM[0] == FaceRecognizer.FINAL_DIM[1]

    '''
    returns the final dimensions entered at the top
    '''
    def final_dimensions(self):
        return self.FINAL_DIM

    '''
    Initializes the racognizer to a new image
    '''
    def new_image(self, image_path):
        try:
            os.remove(self.pic_path)
        except (FileNotFoundError, AssertionError) as e:
            pass

        self.pic_path = image_path


    def new_image_form64(self, b64_string):
        image = Image.open(b64_string)
        path = "FR" + b64_string
        image.save(path, "JPEG")
        self.new_image(path)




    '''
    :return the largest face as a PIL image in size = FINAL_DIM
    :arg upsize, when false, will not allow to resize image to greater than import dimensions
    '''
    def get_largest_face(self, upsize = False):
        assert self.pic_path != "", "Didnt specify file path"

        fr_image = f_r.load_image_file(self.pic_path)
        face_coords = f_r.face_locations(fr_image)

        faces = []
        face_sizes = []
        for coords in face_coords:
            cropped_face = self.get_face_subimage(coords)
            width, height = cropped_face.size
            area = width * height
            faces.append(cropped_face)
            face_sizes.append(area)


        largest_index = FaceRecognizer.index_of_max(face_sizes)
        largest_face = faces[largest_index]

        assert FaceRecognizer.determine_if_valid(face_sizes, largest_index) == True, "Too many faces that are equally likely to be subject"

        if (not upsize):
            assert FaceRecognizer.FINAL_DIM[0] <= largest_face.size[0], "Warning: You are upsizing the image. If you want to avoid this, run again with attr: upsize = True"


        largest_face = largest_face.resize(self.FINAL_DIM)

        return largest_face

    '''
    :return largest face, resized to FINAL_DIM and returned as numpy tensor of ____ * ____ * 3 (R, G, B)
    '''
    def get_face_as_numpy(self):
        image = self.get_largest_face()
        pix = np.array(image)
        return pix

    def get_face_as_base64(self, upsize = False):
        image = self.get_largest_face(upsize)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())

    '''
    returns a fully connected tensor of the image as a np array (reshaping into 1D arr)
    '''
    def get_face_1D_numpy(self):
        nparr= self.get_face_as_numpy()
        assert nparr.size == self.FINAL_DIM[0] * self.FINAL_DIM[1] *3
        return nparr.reshape(nparr.size)

    '''
    
    Returns the facial landmarks of a person's face. For example: 

    {"chin": [[-7, 30], [-4, 51], [0, 73], [6, 92], [14, 110], [24, 126], [37, 141], [54, 152], [72, 155], [90, 151], [103, 140], [112, 123], [120, 106], [128, 86], [133, 67], [134, 44], [133, 23]], 
    "left_eye": [[24, 37], [33, 33], [44, 34], [53, 39], [43, 41], [33, 41]], 
    "right_eyebrow": [[78, 27], [87, 20], [97, 15], [109, 11], [119, 14]], 
    "nose_tip": [[60, 82], [68, 85], [77, 88], [84, 85], [90, 80]], 
    "left_eyebrow": [[9, 17], [22, 13], [35, 14], [49, 18], [60, 25]], 
    "bottom_lip": [[103, 106], [97, 123], [85, 132], [75, 135], [64, 133], [50, 125], [40, 108], [45, 109], [64, 121], [75, 122], [85, 120], [100, 106]], 
    "right_eye": [[84, 38], [91, 31], [101, 30], [109, 33], [102, 38], [93, 39]], 
    "nose_bridge": [[71, 37], [73, 49], [76, 62], [78, 75]], 
    "top_lip": [[40, 108], [52, 100], [65, 97], [75, 99], [85, 96], [96, 98], [103, 106], [100, 106], [85, 104], [76, 105], [65, 104], [45, 109]]}

    if running on multiple threads/processes, be sure to include unique thread/process number when accesing this method:
        This method temporarily saves the file to a hard drive location, if process numbers are not specified,
        multiple threads will try to write to the same location    

    '''

    def get_facial_landmarks(self, thread_num=-1):
        if thread_num == -1:
            file_name = "Temp/Temp.png"
        else:
            file_name = "Temp/Temp" + str(thread_num) + ".png"
        self.save_largest_face(file_name)
        im = f_r.load_image_file(file_name)
        return f_r.face_landmarks(im)

    '''
    Saves the largest image to desired path in JPEG format
    '''
    def save_largest_face(self, path):
        assert self.pic_path != ""

        largest_face = self.get_largest_face()
        largest_face.save(path, "JPEG")


    '''
    Helper method to determine if a pic is a valid profile pic
    '''
    def determine_if_valid(face_sizes, largest_face_index):
        assert len(face_sizes) < 12, "Too many faces in photo! Don't Bother"
        threshold_ratio = 0.75 #if other faces are this larger than this ratio compared to the largest face,
                               #  it is not valid

        largest_face_size = face_sizes.pop(largest_face_index)

        for i in face_sizes:
            if(i > threshold_ratio*largest_face_size):
                return False

        return True

    '''
    Helper method to crop the image
    '''
    def get_face_subimage(self, fr_coords):
        pic = Image.open(self.pic_path)
        top, right, bottom, left = fr_coords
        result = pic.crop((left, top, right, bottom))

        return result


    '''
    Helper method to find largest face in an arr of faces
    '''
    def index_of_max(arr):
        assert len(arr) > 0, "No faces found in image"

        biggest = max(arr)
        for i in range(len(arr)):
            if arr[i] == biggest:
                return i
        return arr[0]



