import cv2
import dlib
import face_recognition as fr
from pprint import pprint
from PIL import Image

class FaceRecognizer:
    def __init__(self):
        self.pic_path = ""

    def new_image(self, image_path):
        self.pic_path = image_path

    def get_largest_face(self):
        assert self.pic_path != ""

        fr_image = fr.load_image_file(self.pic_path)
        face_coords = fr.face_locations(fr_image)

        faces = []
        face_sizes = []
        for coords in face_coords:
            print(coords)
            cropped_face = self.get_face_subimage(coords)
            width, height = cropped_face.size
            area = width * height
            faces.append(cropped_face)
            face_sizes.append(area)

        largest_index = FaceRecognizer.index_of_max(face_sizes)
        largest_face = faces[largest_index]

        return largest_face

    def get_face_subimage(self, fr_coords):
        pic = Image.open(self.pic_path)
        top, right, bottom, left = fr_coords
        result = pic.crop((left, top, right, bottom))

        return result

    def save_largest_face(self, path):
        assert self.pic_path != ""

        largest_face = self.get_largest_face()
        largest_face.save(path, "PNG")


    def index_of_max(arr):
        assert len(arr) > 0

        biggest = max(arr)
        for i in range(len(arr)):
            if arr[i] == biggest:
                return i
        return arr[0]


locations = []

'''
image0 = fr.load_image_file("0.jpg")
locations.append(fr.face_locations(image0))

image1 = fr.load_image_file("1.jpg")
locations.append(fr.face_locations(image1))

image2 = fr.load_image_file("2.jpg")
locations.append(fr.face_locations(image2))

image3 = fr.load_image_file("3.jpg")
locations.append(fr.face_locations(image3))

image4 = fr.load_image_file("4.jpg")
locations.append(fr.face_locations(image4))

image5 = fr.load_image_file("5.jpg")
locations.append(fr.face_locations(image5))

print("... \n")

for i in locations:
    pprint(i)
    print("... \n")
'''


recognizer = FaceRecognizer()
recognizer.new_image("2.jpg")
recognizer.save_largest_face("result.png")

