from sklearn.datasets import load_files
import numpy as np
import os
import cv2
from glob import glob
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import random


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
    dirName = 'static/tempDir'
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    n = random.randint(0, 30000) #force a web browser NOT to cache images
    cropped_img_path =  dirName + '/img' + str(n) + '.jpg'
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
    files = glob('static/tempDir/*')
    for f in files:
        os.remove(f)


###########################################
###  Convert matplotlib figures to png  ###
###########################################

def convert_plt2png(plt_fig):
    """
    Make Matplotlib write to BytesIO file object and grab return the object's string
    Parameters:
        plt_fig: matplotlib figure
    Return:
        figdata_png: extract string (stream of bytes)
    """
    #Matplotlib write the PNG data to the BytesIO buffer
    buf = BytesIO() # create the buffer
    plt_fig.savefig(buf, format='png') # save figure to the buffer
    buf.seek(0) # rewind your buffer
    figdata_png = buf.getvalue()  # extract string (stream of bytes)

    return figdata_png

def embed_png2html(figdata_png):
    """
    Convert the data to base64 format before the PNG data can be embedded in HTML
    Parameters:
        plt_fig: matplotlib figure
    Return:
        figdata_png: base64 encoding images
    """
    figdata_png = base64.b64encode(figdata_png)
    figdata_png = figdata_png.decode('utf-8') # base64.b64encode return byte
    return figdata_png
