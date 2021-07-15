import cv2
import math 

def resize(img, ratio):
    '''
    resize the image while maintaining the aspect ratio
    '''
    h, w = img.shape[:2]
    h /= ratio
    w /= ratio
    dim = (int(w), int(h))
    resize_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resize_img


def resize_to_xmp(img, xmp):
    '''
    resize to x megapixel 
    if image is less than 1mp it will return the same image
    '''
    h, w = img.shape[:2]
    if w*h < 1000000 * xmp:
        return img
    ratio = math.sqrt((xmp*1000000) / (w * h))
    return resize(img, ratio)
