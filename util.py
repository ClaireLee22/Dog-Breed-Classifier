from sklearn.datasets import load_files
import numpy as np
import os
import cv2
from glob import glob
import matplotlib.pyplot as plt

#########################################
###  Handle cropped images functions  ###
#########################################
def crop_detected_faces(BGR_img, faces):
    """
    Crop faces on the test human image
    Parameters:
        BGR_img: a color (BGR) image has been read by OpenCV
        faces: number of faces that have been detected in an image
    Return:
        cropped_imgs: cropped images with detected faces
    """
    cropped_imgs = []
    offset = 25
    height, width = BGR_img.shape[:2]
    for i in range(len(faces)):
        x,y,w,h = faces[i]
        cropped_img = BGR_img[y-offset : y+h+offset, x-offset : x+w+offset]
        if(y-offset < 0):
            cropped_img = BGR_img[0 : y+h+offset, x-offset : x+w+offset]
        if(x-offset < 0):
            cropped_img = BGR_img[y-offset : y+h+offset, 0 : x+w+offset]
        if(y+h+offset > height):
            cropped_img = BGR_img[y-offset : height, 0 : x+w+offset]
        if(x+w+offset > width):
            cropped_img = BGR_img[y-offset : y+h+offset, 0 : width]
        cropped_imgs.append(cropped_img)
    return cropped_imgs

def save_cropped_imgs(num, cropped_imgs):
    """
    Save cropped images to temp directory for later use
    Parameters:
        cropped_imgs: cropped images with detected faces
        num: number of cropped images from a test image
    Return:
        cropped_img_path: a string-valued file path to a cropped image
    """
    dirName = 'tempDir'
    if not os.path.exists(dirName):
        os.mkdir(dirName)

    cropped_img_path =  dirName + '/img' + str(num) + '.jpg'
    cv2.imwrite(cropped_img_path, cropped_imgs[num])
    return cropped_img_path

def delete_cropped_images():
    """
    Delete all files in temp directory
    Parameters:
        None
    Return:
        None
    """
    files = glob('tempDir/*')
    for f in files:
        os.remove(f)


############################
###  show upload image   ###
############################
def show_upload_image(img_path):
    """
    Display the test image on predicted result
    Parameters:
        img_path: a string-valued file path to an image
    Return:
        None
    """
    img = cv2.imread(img_path)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
