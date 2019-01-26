from dog_names import dog_names
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing import image
from extract_bottleneck_features import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class InceptionV3:
    def __init__(self, img_path, InceptionV3_model=None, input_shape=(5,5,2048)):
        self.img_path = img_path
        if InceptionV3_model is None:
            model = Sequential()
            model.add(GlobalAveragePooling2D(input_shape = input_shape))
            model.add(Dense(133, activation='softmax'))
            model.load_weights('saved_models/weights.best.InceptionV3.hdf5')
            InceptionV3_model = model
        self.InceptionV3_model = InceptionV3_model
        self.input_shape = input_shape


    def path_to_tensor(self):
        """
        Convert a color image to a 4D tensor with shape (1, 224, 224, 3) to supply Keras CNN
        Parameters:
            img_path: a string-valued file path to an image
        Return:
            a 4D tensor suitable with shape (1, 224, 224, 3)
        """
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(self.img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    def predict_top5_breeds(self):
        """
        Use InceptionV3 model to predicte dog breed and return top5 predicted breeds
        Parameters:
            None
        Return:
            top5_breeds_idx: top5 dog breed idx that is predicted by the model
            top5_breeds_prob: top5 dog breed probabilities that are corresponding to top5 dog breed idx
        """
        # extract bottleneck features
        bottleneck_feature = extract_InceptionV3(self.path_to_tensor())
        # obtain predicted vector
        predicted_vector = self.InceptionV3_model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        # np.argsort(-predicted_vector) return matrix shape (1, 133)
        top5_breeds_idx= np.argsort(-predicted_vector)[0][:5] # return top5 probability class idx
        top5_breeds_prob= [predicted_vector[0][idx] for idx in top5_breeds_idx]# return top5 probability
        return top5_breeds_idx, top5_breeds_prob

    def isMix_mutt(self, top5_breeds_prob, top5_labels):
        """
        Decide whether if the detected dog/human is resemble to mix mutt dog breeds or not.
        If the difference of top 1 probability and top 2 probability is less than 0.1, it will be classified as
        mix mutt.
        Parameters:
            top5_breeds_prob: top5 dog breed probabilities that is predicted by the model
            top5_labels: top5 dog breed labels that are corresponding to top5 dog breed idx
        Return:
            None
        """
        if (top5_breeds_prob[0] - top5_breeds_prob[1] <= 0.1):
            print("You look like", top5_labels[0], '&', top5_labels[1] ,"mixed breed dog!")
        else:
            print("You look like", top5_labels[0], "!")

    def show_top5_result(self):
        """
        Visualize top 5 dog breeds that are predicted by the model via bar chart
        Parameters:
            None
        Return:
            None
        """
        top5_breeds_idx, top5_breeds_prob = self.predict_top5_breeds()
        top5_labels = [dog_names[idx] for idx in top5_breeds_idx]
        self.isMix_mutt(top5_breeds_prob, top5_labels)

        #plot top5 result
        plt.subplot(1,2,1)
        plt.barh(top5_labels, top5_breeds_prob)
        plt.xlabel('probability')
        plt.xlim(0, 1.0)
        plt.title('dog breeds classifer')
        plt.show()
