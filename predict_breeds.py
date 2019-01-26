import dog_names as dn
from dog_filters import Dog_filters as df
from InceptionV3 import InceptionV3 as iV3
from ResNet50 import ResNet50 as rn50
import util as u

import numpy as np
import tensorflow as tf
import cv2
from keras import backend as K


class Predict_breeds:
    def __init__(self, img_path):
        self.img_path = img_path


    def Improved_OpenCV_face_detector(self):
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


    def predict_breed_for_human_only(self):
        """
        Predict which dog breeds the cropped image is resemble.
        cropped image includes only one detected face, so no need to run the whole process in improved_predict_breed function
        Parameters:
            img_path: a string-valued file path to an image
        Return:
            None
        """
        iV3_model = iV3(self.img_path)
        dog_filters = df(self.img_path)
        faces, BGR_img = self.Improved_OpenCV_face_detector()
        print('Hello, human!')
        dog_filters.apply_snapchat_filter()
        iV3_model.show_top5_result()


    def detect_face_on_cropped_imgs(self, cropped_imgs):
        """
        Detect whether if any faces in an image
        Parameters:
            img_path: a string-valued file path to an image
        Return:
            True: if face is detected in image stored at img_path
            False: if face is NOT detected in image stored at img_path
        """
        for i in range(len(cropped_imgs)):
            cropped_img_path = u.save_cropped_imgs(i, cropped_imgs)
            self.img_path = cropped_img_path
            self.predict_breed_for_human_only()


    def process_predict(self):
        """
        Predict which dog breeds the image is resemble.
        if dog and face are detected on the same image
            - predict dog breeds on the detected dog
            - crop the detected face area on the image and predict dog breeds on the detected face

        if more than one detected faces on the same image
            - crop the detected face area on the image and predict dog breeds on the individual detected face

        if only one face is detected on the same image
            - predict dog breeds on the detected face

        if no dog and human are detected
            - print "No human. No dog."

        Parameters:
            None
        Return:
            None
        """
        rn50_model = rn50()
        iV3_model = iV3(self.img_path)
        dog_filters = df(self.img_path)
        faces, BGR_img = self.Improved_OpenCV_face_detector()
        dogs =  rn50_model.dog_detector(self.img_path)
        #if dog and human in the same image, model predicts dog breeds will always based on the dog
        #so we have to cropped the human image from the dog
        if(dogs != 0):
            print('Hello, dog!')
            u.show_upload_image(self.img_path)
            iV3_model.show_top5_result()
            if(len(faces) > 0):
                cropped_imgs = u.crop_detected_faces(BGR_img, faces)
                self.detect_face_on_cropped_imgs(cropped_imgs)
                u.delete_cropped_images()
        #if more than one people in the same image, model predicts dog breeds will always show one result
        #so we have to crop the human image to individuals
        else:
            if(len(faces) > 1):
                cropped_imgs = u.crop_detected_faces(BGR_img, faces)
                self.detect_face_on_cropped_imgs(cropped_imgs)
                u.delete_cropped_images()
            elif(len(faces) == 1):
                print('Hello, human!')
                dog_filters.apply_snapchat_filter()
                iV3_model.show_top5_result()
            else:
                print('No human. No dog.')
                u.show_test_image(self.img_path)
