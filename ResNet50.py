from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import tensorflow as tf


ResNet50_model = ResNet50(weights='imagenet')

class ResNet50:
    def __init__(self):
        pass

    def path_to_tensor(self, img_path):
        """
        Convert a color image to a 4D tensor with shape (1, 224, 224, 3) to supply Keras CNN
        Parameters:
            img_path: a string-valued file path to an image
        Return:
            a 4D tensor suitable with shape (1, 224, 224, 3)
        """
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)


    def ResNet50_predict_labels(self, img_path):
        """
        Use the model to extract the predictions and get the highest predicted object classes
        Parameters:
            img_path: a string-valued file path to an image
        Return:
            an integer corresponding to the model's predicted object class
        """
        # returns prediction vector for image located at img_path
        img = preprocess_input(self.path_to_tensor(img_path))
        return np.argmax(ResNet50_model.predict(img))


    def dog_detector(self, img_path):
        """
        Detect whether if any dogs in the images
        Parameters:
            img_path: a string-valued file path to an image
        Return:
            True: if a dog is detected in an image
            False: if a dog is NOT detected in an image
        """
        prediction = self.ResNet50_predict_labels(img_path)
        return ((prediction <= 268) & (prediction >= 151))
