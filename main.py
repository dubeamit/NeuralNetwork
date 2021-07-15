import cv2
import numpy as np


def preprocess(img):
    img = img / 255
    return img.flatten()

def sigmoid(x):
    return 1 /(1 + np.exp(-x))

def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))

def feedforward(input, weights):
    convolution = np.dot(input, weights)
    output = sigmoid(convolution)
    return output

def backpropogation(output, label, input):
    error = output - label
    adjustment = error * sigmoid_d(output)
    weights_adjustment = np.dot(adjustment, input)
    return weights_adjustment


# 1. read image
horizontal_img = cv2.imread('images/horizontal.png', 0)
horizontal_img = preprocess(cv2.resize(horizontal_img, (3,3), interpolation=0))
vertical_img = cv2.imread('images/vertical.png', 0)
vertical_img = preprocess(cv2.resize(vertical_img, (3,3), interpolation=0))

# print(vertical_img)
# print(horizontal_img)

dataset = np.array([vertical_img, horizontal_img])
labels = np.array([0, 1])

# 2. weights
weights = np.array([0.5]*9)
# print(weights.shape)

epochs = 100
for i in range(epochs):
    output = feedforward(dataset, weights)
    weights -= backpropogation(output, labels, dataset)
    # print(weights)
    
print(output)
