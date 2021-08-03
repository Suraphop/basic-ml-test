#temp
import cv2
import numpy as np
import pandas as pd
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import time

DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

result = 'init'

def roi(frame):
    height, width, channels = frame.shape
    y =  height // 2 -150
    x = width // 2 -200
    h = 300
    w = 400
    frame = frame[y:y+h, x:x+w]
    return frame

def searchingArea(frame):
    height, width, channels = frame.shape
    #monitor area
    upper_left_monit = (width // 2 -120 , height // 2 -70)
    bottom_right_monit = (width // 2 +120 , height // 2 +70)
    cv2.rectangle(frame, upper_left_monit, bottom_right_monit, (255, 0, 0), thickness=2)
    cv2.putText(frame, 'temp here!!', upper_left_monit, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
    #temp number area
    upper_left = (width // 2 -200 , height // 2 -150)
    bottom_right = (width // 2 +200 , height // 2 +150)
    cv2.rectangle(frame, upper_left, bottom_right, (255, 0, 0), thickness=1)
    #cv2.putText(frame, 'move to fit area', (5,30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
    return frame

def readTemp(frame):
    #Canny
    image = imutils.resize(frame, height=400,width =300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 20, 100, 255)
    #find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # the thermostat display
        if len(approx) == 4:
            displayCnt = approx
            break
    #warped
    if displayCnt is not None:
        warped = four_point_transform(gray, displayCnt.reshape(4, 2))
        #output = four_point_transform(image, displayCnt.reshape(4, 2))

        #threh
        thresh = cv2.threshold(warped, 50, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 9)) # dark
        #thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) #dark

        ksize = (3,3) #light
        # ksize = (5,5) #dark
        kernels = np.ones(ksize, np.uint8)
        thresh = cv2.dilate(thresh, kernels, iterations=1) #การพองตัว
        thresh = thresh[20:-20,30:-20]

        #split contours
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	    cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        #print(len(cnts))
        digitCnts = []
        # loop over the digit area candidates
        for c in cnts:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            # if the contour is sufficiently large, it must be a digit
            # print('w',w)
            # print('h',h)
            if w >= 30 and (h >= 30 and h <= 130):
                digitCnts.append(c)
        #print('find digit :',len(digitCnts))

        if len(digitCnts) != 3:
            #print('detect digit :',len(digitCnts))
            return
        
        digitCnts = contours.sort_contours(digitCnts,
	    method="left-to-right")[0]
        digits = []

        for c in digitCnts:
            # extract the digit ROI
            (x, y, w, h) = cv2.boundingRect(c)
            roi = thresh[y:y + h, x:x + w]

            # compute the width and height of each of the 7 segments
            # we are going to examine
            (roiH, roiW) = roi.shape
            (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
            dHC = int(roiH * 0.05)
            # define the set of 7 segments
            segments = [
                ((0, 0), (w, dH)),	# top
                ((0, 0), (dW, h // 2)),	# top-left
                ((w - dW, 0), (w, h // 2)),	# top-right
                ((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
                ((0, h // 2), (dW, h)),	# bottom-left
                ((w - dW, h // 2), (w, h)),	# bottom-right
                ((0, h - dH), (w, h))	# bottom
            ]
            on = [0] * len(segments)

                # loop over the segments
            for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                # extract the segment ROI, count the total number of
                # thresholded pixels in the segment, and then compute
                # the area of the segment
                segROI = roi[yA:yB, xA:xB]
                total = cv2.countNonZero(segROI)
                area = (xB - xA) * (yB - yA)
                # if the total number of non-zero pixels is greater than
                # 50% of the area, mark the segment as "on"
                if total / float(area) > 0.5:
                    on[i]= 1
            # lookup the digit and draw it on the image
            try:
                global result
                digit = DIGITS_LOOKUP[tuple(on)]
                digits.append(digit)
                if len(digits) == 3:
                    result = "{}{}.{}".format(*digits)
  
            except:
              return

            # cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # cv2.putText(output, str(digit), (x - 10, y - 10),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    #return result

cap = cv2.VideoCapture(1)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        readTemp(frame)
        text = 'Your temp is '+result
        cv2.putText(frame, text, (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow('frame', searchingArea(frame))
        result='.'
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()