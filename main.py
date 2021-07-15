from augmentation import resize
from os import error
import cv2
import numpy as np


def preprocess(img):
    img = img / 255
    return img.flatten()

def sigmoid(x):
    return 1 /(1 + np.exp(-x))

def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))


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
# print(weights)

epochs = 100
for i in range(epochs):
    print()
    print('round', i+1)
    temp_weigths = weights.copy()
    for img, label in zip(dataset, labels):
        # 3. convolve 
        convolution = sum(img * weights)
        print('convolution', convolution)
        # 4. activation function
        result = sigmoid(convolution)
        print('result', result)
        # 5. error
        error = result - label
        print('error', error)
        # 6. Adjustment (slope)
        adjustment = error * sigmoid_d(result)
        print('adjustment', adjustment)
        # 7. update weights
        temp_weigths -= np.dot(img, adjustment)
        print('temp_weights', temp_weigths)
        print()

    # update weight after every round in epochs
    weights = temp_weigths.copy()
    print('weights', weights)

