import cv2
import numpy as np


def preprocess(img):
    img = img / 255
    return img.flatten()


# 1. read image
horizontal_img = cv2.imread('images/horizontal.png', 0)
horizontal_img = preprocess(cv2.resize(horizontal_img, (3,3), interpolation=0))
vertical_img = cv2.imread('images/vertical.png', 0)
vertical_img = preprocess(cv2.resize(vertical_img, (3,3), interpolation=0))

# print(vertical_img)
# print(horizontal_img)

dataset = np.array([vertical_img, horizontal_img])
labels = np.array([[0, 1]]).T



class Layers:
    def __init__(self, input_size, output_size):
        print('Initialzing layers')
        self.weights = np.random.rand(input_size, output_size) # output between 0-1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_d(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def feedword(self, input):
        convolution = np.dot(input, self.weights)
        output = self.sigmoid(convolution)
        return output

    def backpropogation(self, output, labels, input):
        error = output - labels
        adjustment = error * self.sigmoid_d(output)
        weights_adjustment = np.dot(input.T, adjustment)
        self.weights -= weights_adjustment

        
# create the layers
output_layer = Layers(9, 1)
epochs = 100
for epoch in range(epochs):
    output = output_layer.feedword(dataset)
    output_layer.backpropogation(output, labels, dataset)

print('final weights', output_layer.weights)
print('ouput', output)
