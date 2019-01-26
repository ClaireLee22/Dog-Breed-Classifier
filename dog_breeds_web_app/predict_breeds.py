import dog_names as dn
from dog_filters import Dog_filters as df
from InceptionV3 import InceptionV3 as iV3
import util as u

import numpy as np
import tensorflow as tf
import cv2
from keras import backend as K


class Predict_breeds:
    def __init__(self, img_path):
        self.img_path = img_path

    def load_model(self):
        iV3_model = iV3()
        return iV3_model

    def OpenCV_face_detector(self):
        """
        Detect whether if any faces in an image
        Parameters:
            None
        Return:
            faces: number of faces that have been detected in an image
            img: a color (BGR) image has been read by OpenCV
        """
        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
        img = cv2.imread(self.img_path) #BGR image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        return faces, img


    def process_predict(self):
        """
        Predict which dog breeds the image is resemble.
        Parameters:
            None
        Return:
            None
        """
        iV3_model = self.load_model();
        faces, BGR_img = self.OpenCV_face_detector()
        filter_img_paths = []
        figdata_pngs = []
        isHumanOrDogs = []
        pred_messages= []

        #if dog and human in the same image, model predicts dog breeds will always based on the dog
        #so we have to cropped the human image from the dog
        if(len(faces) > 0):
            cropped_imgs = u.crop_detected_faces(BGR_img, faces)
            for i in range(len(cropped_imgs)):
                isHumanOrDog = 'Hello, human!'
                cropped_img_path = u.save_cropped_imgs(i, cropped_imgs)
                dog_filters = df(cropped_img_path)
                figdata_png, pred_message= iV3_model.show_top5_result(dog_filters.face_img_path)
                dog_filters.apply_snapchat_filter()

                filter_img_paths.append(dog_filters.face_img_path)
                figdata_pngs.append(figdata_png)
                isHumanOrDogs.append(isHumanOrDog)
                pred_messages.append(pred_message)
        else:
            isHumanOrDog = 'Hello, dog!'
            filter_img_path = None
            figdata_png, pred_message = iV3_model.show_top5_result(self.img_path)

            filter_img_paths.append(filter_img_path)
            figdata_pngs.append(figdata_png)
            isHumanOrDogs.append(isHumanOrDog)
            pred_messages.append(pred_message)

        return filter_img_paths, figdata_pngs, isHumanOrDogs, pred_messages


def play_dog_breeds(img_path):
    u.delete_cropped_images() # empty the tempDir for each round
    #TypeError: Cannot interpret feed_dict key as Tensor
    #solution: add  K.clear_session before and after prediction
    K.clear_session()
    p = Predict_breeds(img_path)
    K.clear_session()
    return p.process_predict()
