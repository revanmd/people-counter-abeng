import cv2
import numpy as np

import imutils
import math
from imutils.object_detection import non_max_suppression
from imutils import paths

from preprocess import *
from interface import *
from hog import *
from svm import *
from helpers import *

if __name__ == "__main__":
   # PARAMETERS
    args = create_parser()
    video = args['video']
    roi_mode = args['roi creation mode']
   
    # CONSTANTS
    videopath,__=video.split(".",-1)
    __,videoname=videopath.split('/',-1)
    width = 800
    height= 640
    imageset=[]
    counter,textIn,textOut = 0,0,0
   
    # CLASS SVM
    svm = SVMClassifier()
    
    # CLASS HOG
    hog = HOG()
    hog.set_winstride((4, 4))
    hog.set_padding((8, 8))
    hog.set_scale(1.1)
    hog.set_descriptor(svm.get_classifier())
     
    # DEFINING ROI
    camera = cv2.VideoCapture(video)
    grabbed, frame = camera.read()
    
    roi = define_roi(roi_mode, frame, videopath)
    frame = resize_frame(frame)
    gray = gray_frame(frame)
    gray = blur_frame(gray)
    
    cv2.destroyWindow('ROI selector')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vid_out = cv2.VideoWriter('Output_Video/'+videoname+'.avi', fourcc, 20.0, (int(camera.get(3)), int(camera.get(4))), isColor=True)
    
    # loop over the frames of the video
    st=0.0
    cs=[0,0]
    tc,f,tb,cts=0,0,0,0
    while (camera.isOpened()):
        grabbed, frame = camera.read()  
        
        if not grabbed:
            break
        
        frame = resize_frame(frame)
        forig = frame
        frame = frame[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]
        gray  = gray_frame(frame)
        gray  = blur_frame(gray)
        
        ori=frame.copy()
        (rects, weights) = hog.detectMultiscale(gray)
        
        for (a, b, c, d) in rects:
            cv2.rectangle(ori, (a, b), (a + c, b + d), (255,0, 0), 0)
            
        rects = np.array([[a, b, a + c, b + d] for (a, b, c, d) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.7)
        
        for (xA, yA, xB, yB) in pick:
            rect=cv2.rectangle(frame, (xA, yA), (xB, yB), (255,0,0), 2)
            rectf=frame[yA:yA+yB,xA:xA+xB]
            imageset2=np.asarray(imageset)
            xc=(xA + xB) /2
            yc=(yA + yB) /2
            rectangleCenterPoint = (int(xc), int(yc))
            cv2.circle(frame, rectangleCenterPoint, 1, (255,0,0), 2)
            
            if counter==0:
                counter+=1
                pimg='Output_Images/'+videoname+'_person'+str(counter)+'.jpg'
                img=cv2.imwrite(pimg,rectf)
                imageset.append(rectf)
                continue
            u=0
            
            if cts==3:
                if cs[counter]>0:
                    textIn+=1
                    f=1
                elif cs[counter]<0:
                    textOut+=1
                    f=-1	
                cts+=1

            if (cts>=5) and (cs[counter]*f)<0:
                if cs[counter]>0:
                    textIn+=1
                    # textOut-=1
                    f=1
                elif cs[counter]<0:
                    textOut+=1
                    # textIn-=1 
                    f=-1  
                    
                      	    
            for t in imageset2:
                if feature_matching(rectf,t)>2:
                    u=0
                    break
                else:
                    u+=1


            if u==0:
                if yc>tc and yB>tb:
                    cs[counter]+=1
                elif yc<tc and yB<tb:
                    cs[counter]-=1
                tc=yc
                tb=yB
                cts+=1
                
            if u>0:  
                cs.append(0)
                cts=0
                counter+=1
                pimg='Output_Images/'+videoname+'_person'+str(counter)+'.jpg'
                img=cv2.imwrite(pimg,rectf)
                imageset.append(rectf)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        display_output(forig, roi, textIn, textOut)
        vid_out.write(forig)

    # cleanup the camera and close any open windows
    camera.release()
    vid_out.release()
    cv2.destroyAllWindows()
    print("No. of images created=",counter)
    if roi_mode=='manually':
        print("Do you wish to save the created roi to be used as an optional pre-tested file for this video next run onwards (if it gave good results)")
        wr=create_roi(videoname,roi)
