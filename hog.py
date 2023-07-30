import cv2

class HOG:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.winstride = (0,0)
        self.padding = (0,0)
        self.scale = 1
        
    def set_winstride(self, winstride):
        self.winstride = winstride
        
    def set_padding(self, padding):
         self.padding = padding
    
    def set_scale(self, scale):
        self.scale = scale
        
    def set_descriptor(self, descriptor):
        self.hog.setSVMDetector(descriptor)
    
    def detectMultiscale(self, frame):
        return self.hog.detectMultiScale(frame, winStride=self.winstride ,padding=self.padding, scale=self.scale)    