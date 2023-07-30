import cv2

class SVMClassifier:
    def __init__(self):
        # Ini adalah pretrained HOG People Detector untuk SVM Paramter
        self.SVMClassifier = cv2.HOGDescriptor_getDefaultPeopleDetector()
    
    def get_classifier(self):
        return self.SVMClassifier