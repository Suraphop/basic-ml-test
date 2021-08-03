
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2

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


import numpy as np

image = cv2.imread("oxy4.jpg") # 7,10,11
# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, height=500,width =200)
print(image.shape)

image_cvt = image.copy()

hsv = cv2.cvtColor(image_cvt, cv2.COLOR_BGR2HSV)
lb = np.array([10, 50, 50])
ub = np.array([40, 250, 250])
mask = cv2.inRange(hsv, lb, ub)
image_cvt = cv2.bitwise_and(image_cvt, image_cvt, mask=mask)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_cvt = cv2.cvtColor(image_cvt, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray_cvt, (5, 5), 0)
edged = cv2.Canny(blurred, 150, 150, 255) #220 220



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
	# if the contour has four vertices, then we have found
	# the thermostat display
	if len(approx) == 4:
		displayCnt = approx
		break
print('len:',len(displayCnt))


warped = four_point_transform(gray, displayCnt.reshape(4, 2))
output = four_point_transform(image, displayCnt.reshape(4, 2))



# threshold the warped image, then apply a series of morphological
# operations to cleanup the thresholded image
import numpy as np
thresh = cv2.threshold(warped, 100, 250,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 1)) # dark
#thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) #dark

# ksize = (3,3) #light
# kernels = np.ones(ksize, np.uint8)

# thresh = cv2.dilate(thresh, kernels, iterations=1) #การพองตัว

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(1, 2 * 4))
thresh = cv2.dilate(thresh, kernel)
thresh2 = thresh.copy()


y,x = thresh.shape

yy_0 = int(y//5)
yy = int(y//1.6)
xx_0 = int(y//5)
xx = int(y//1.6)

thresh = thresh[yy_0:yy,xx_0:xx]


yy_0_2 = int(y//1.7)
xx_0_2 = int(y//5)
xx_2 = int(y//1.4)

thresh2 = thresh2[yy_0_2:,xx_0_2:xx_2]



# find contours in the thresholded image, then initialize the
# digit contours lists
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print(len(cnts))
digitCnts = []
# loop over the digit area candidates
for c in cnts:
	# compute the bounding box of the contour
	(x, y, w, h) = cv2.boundingRect(c)
	# if the contour is sufficiently large, it must be a digit
	# print('w',w)
	# print('h',h)
	if w >= 10 and (h >= 5 and h <= 150):
		digitCnts.append(c)
print(len(digitCnts))


# sort the contours from left-to-right, then initialize the
# actual digits themselves
digitCnts = contours.sort_contours(digitCnts,
	method="left-to-right")[0]
digits = []


# loop over each of the digits
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
	digit = DIGITS_LOOKUP[tuple(on)]
	digits.append(digit)
	cv2.rectangle(output, (x+xx_0, y+yy_0), (x + xx, y + yy), (0, 255, 0), 1)
	cv2.putText(output, str(digit), (x - 10, y - 10),
	cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)


number = "{}{}".format(*digits)

cv2.putText(image, "Oxygen :"+str(number), (10, 20),
	cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
print(number)


# find contours in the thresholded image, then initialize the
# digit contours lists
cnts = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print(len(cnts))
digitCnts = []
# loop over the digit area candidates
for c in cnts:
	# compute the bounding box of the contour
	(x, y, w, h) = cv2.boundingRect(c)
	# if the contour is sufficiently large, it must be a digit
	# print('w',w)
	# print('h',h)
	if w >= 10 and (h >= 10 and h <= 130):
		digitCnts.append(c)



# sort the contours from left-to-right, then initialize the
# actual digits themselves
digitCnts = contours.sort_contours(digitCnts,
	method="left-to-right")[0]
digits = []


# loop over each of the digits
for c in digitCnts:
	# extract the digit ROI
	(x, y, w, h) = cv2.boundingRect(c)
	roi = thresh2[y:y + h, x:x + w]

	
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
	digit = DIGITS_LOOKUP[tuple(on)]
	digits.append(digit)
	cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
	cv2.putText(output, str(digit), (x - 10, y - 10),
	cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)


number = "{}{}".format(*digits)

cv2.putText(image, "pluse :"+str(number), (10, 50),
	cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
print(number)


cv2.imshow("Input", image)
#cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()


