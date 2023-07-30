import argparse
import cv2
import datetime

def create_parser():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-vid','--video',required=True,default="Input/2.mp4",help="Video File Path")
    parser.add_argument('-roi','--roi creation mode',required=False,default="manually",help="Create region of interest-do it 'manually'," + 
                        "or use the 'pre-tested' one which gives good results")
    args = vars(parser.parse_args())
    
    return args

def define_roi(roi_mode, frame, videopath):
    if roi_mode=='manually':
        roi = cv2.selectROI(frame,showCrosshair=False)
    elif roi_mode=='pre-tested':
        try:
            roi_file=open(videopath+'_pre-testedROI.txt','r')
            rois=roi_file.read()
            rois=rois[1:-1]
            roi=rois.split(", ")
            for i in range(len(roi)): 
                roi[i]=int(roi[i])
        except:
            print("The pre-tested Region of Interest file does not exist yet. Please create it manually.")
            roi = cv2.selectROI(frame,showCrosshair=False)
            
    return roi

def display_output(forig, roi, textIn, textOut):
    cv2.rectangle(forig, (roi[0], roi[1]), (roi[0]+roi[2],roi[1]+roi[3]), (0, 255, 0),1)
    cv2.putText(forig, "Total: {}".format(str(textIn + textOut)), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(forig, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, forig.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.imshow("Security Feed", forig)