import cv2
import matplotlib.pyplot as plt
import util as u

dog_left_ear  = cv2.imread('static/filters/dog_left_ear.png', cv2.IMREAD_UNCHANGED) # rgba
dog_right_ear = cv2.imread('static/filters/dog_right_ear.png',cv2.IMREAD_UNCHANGED)
dog_nose = cv2.imread('static/filters/dog_nose.png', cv2.IMREAD_UNCHANGED)
dog_tongue = cv2.imread('static/filters/dog_tongue.png', cv2.IMREAD_UNCHANGED)

dog_filters = [dog_left_ear, dog_right_ear, dog_nose, dog_tongue]

class Dog_filters:
    def __init__(self, face_img_path, scale=0.3):
        self.face_img_path = face_img_path
        self.face_img = cv2.imread(self.face_img_path) # load color (BGR) image
        self.scale = scale
        self.resize_filters=[]


    def resize_filter(self, w):
        """
        Adjust the size of filter images according to the size of the detected face on the test image
        Parameters:
            img: images that has read by OpenCV (BGR image)
            w: width of bounding box for the detected face
        Return:
            img: resized image
        """
        for dog_filter in dog_filters:
            new_w = int(w*0.3) # resize the image to 0.3 scale of the width of the face detector
            new_h = int(dog_filter.shape[0]/dog_filter.shape[1]*new_w) # keep aspect ratio
            resize_filter = cv2.resize(dog_filter, (new_w, new_h))
            self.resize_filters.append(resize_filter)


    def put_dog_left_ear(self, x, y):
        """
        Overlay dog_left_ear filter image to the detected faces on human images
        Parameters:
            dog_left_ear_img: filter image with dog left ear
            human_img: input image for the classifier
            x, y : coordinates of the bounding box for the detected face at the top right corner
            w: width of bounding box for the detected face
            h: height of bounding box for the detected face
        Return:
            human_img: human image with dog left ear filter
        """
        dog_left_ear_filter = self.resize_filters[0]

        #overlay range
        yo = dog_left_ear_filter.shape[0]
        xo = dog_left_ear_filter.shape[1]
        #loop every pixel
        for j in range(yo):
            for i in range(xo):
                for k in range(3):
                    alpha = float(dog_left_ear_filter[j][i][3]/255.0) # read the alpha channel
                    #self.face_img[x + i][y + j][k] = dog_left_ear_filter[i][j][k] # with black background if w/o alpha
                    self.face_img[y+j][x+i][k] = alpha*dog_left_ear_filter[j][i][k]+(1-alpha)*self.face_img[y+j][x+i][k]



    def put_dog_right_ear(self, x, y, w, h):
        """
        Overlay dog_right_ear filter image to the detected faces on human images
        Parameters:
            dog_right_ear_img: filter image with dog right ear
            human_img: input image for the classifier
            x, y : coordinates of the bounding box for the detected face at the top right corner
            w: width of bounding box for the detected face
            h: height of bounding box for the detected face
        Return:
            human_img: human image with dog right ear filter
        """
        dog_right_ear_filter = self.resize_filters[1]

        #overlay range
        yo = dog_right_ear_filter.shape[0]
        xo = dog_right_ear_filter.shape[1]
        #loop every pixel
        for j in range(yo):
            for i in range(xo):
                for k in range(3):
                    alpha = float(dog_right_ear_filter[j][i][3]/255.0) # read the alpha channel
                    self.face_img[y+j][x+w-xo+i][k] = alpha*dog_right_ear_filter[j][i][k]+(1-alpha)*self.face_img[y+j][x+w-xo+i][k]




    def put_dog_nose(self, x, y, w, h):
        """
        Overlay dog_nose filter image to the detected faces on human images
        Parameters:
            dog_nose_img: filter image with dog nose
            human_img: input image for the classifier
            x, y : coordinates of the bounding box for the detected face at the top right corner
            w: width of bounding box for the detected face
            h: height of bounding box for the detected face
        Return:
            human_img: human image with dog nose filter
        """
        dog_nose_filter = self.resize_filters[2]
        #overlay range
        yo = dog_nose_filter.shape[0]
        xo = dog_nose_filter.shape[1]
        #loop every pixel
        for j in range(yo):
            for i in range(xo):
                for k in range(3):
                    alpha = float(dog_nose_filter[j][i][3]/255.0) # read the alpha channel
                    self.face_img[y+int(h/2)+j][x+int((w-xo)/2)+i][k] = alpha*dog_nose_filter[j][i][k]+\
                                                            (1-alpha)*self.face_img[y+int(h/2)+j][x+int((w-xo)/2)+i][k]




    def put_dog_tongue(self, x, y, w, h):
        """
        Overlay dog_tongue filter image to the detected faces on human images
        Parameters:
            dog_tongue_img: filter image with dog tongue
            human_img: input image for the classifier
            x, y : coordinates of the bounding box for the detected face at the top right corner
            w: width of bounding box for the detected face
            h: height of bounding box for the detected face
        Return:
            human_img: human image with dog tongue filter
        """
        dog_nose_filter = self.resize_filters[3]
        #overlay range
        yo = dog_nose_filter.shape[0]
        xo = dog_nose_filter.shape[1]
        #loop every pixel
        for j in range(yo):
            for i in range(xo):
                for k in range(3):
                    alpha = float(dog_nose_filter[j][i][3]/255.0) # read the alpha channel
                    self.face_img[y+h-yo+j][x+int((w-xo)/2)+i][k] = alpha*dog_nose_filter[j][i][k]+\
                                                            (1-alpha)*self.face_img[y+h-yo+j][x+int((w-xo)/2)+i][k]


    def apply_snapchat_filter(self):
        """
        Overlay  dog filter images(left_ear, right_ear, nose, tongue) to the detected faces on human images
        Parameters:
            img_path: a string-valued file path to an image
        Return:
            None
        """
        # extract pre-trained face detector
        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

        # convert BGR image to grayscale
        gray = cv2.cvtColor(self.face_img, cv2.COLOR_BGR2GRAY)

        # find faces in image
        face = face_cascade.detectMultiScale(gray)

        # add overlayer
        for (x,y,w,h) in face:
            self.resize_filter(w)
            self.put_dog_left_ear(x, y)
            self.put_dog_right_ear(x, y, w, h)
            self.put_dog_nose(x, y, w, h)
            self.put_dog_tongue(x, y, w, h)
            cv2.imwrite(self.face_img_path, self.face_img) #replace originak cropped image with filter one
