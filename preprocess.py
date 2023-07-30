import cv2
import imutils

def resize_frame(frame):
    return imutils.resize(frame)

def gray_frame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def blur_frame(frame):
    return cv2.GaussianBlur(frame, (21, 21), 0)