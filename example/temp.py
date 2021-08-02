# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
# define the dictionary of digit segments so we can identify
# each digit on the thermostat
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


# %%
# load the example image

image = cv2.imread("sample11.jpg") # 7,10,11
# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, height=500,width =400)
print(image.shape)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 20, 100, 255)

cv2.imshow('edge',edged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%
# find contours in the edge map, then sort them by their
# size in descending order
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


# %%
# extract the thermostat display, apply a perspective transform
# to it
warped = four_point_transform(gray, displayCnt.reshape(4, 2))
output = four_point_transform(image, displayCnt.reshape(4, 2))


cv2.imshow('d',warped)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%
# threshold the warped image, then apply a series of morphological
# operations to cleanup the thresholded image
import numpy as np
thresh = cv2.threshold(warped, 50, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 9)) # dark
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) #dark


#ksize = (3,3) #light
ksize = (5,5) #dark
kernels = np.ones(ksize, np.uint8)

thresh = cv2.dilate(thresh, kernels, iterations=1) #การพองตัว
offset = 20

thresh = thresh[offset:-20,:]
cv2.imshow('d',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%

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
	if w >= 30 and (h >= 30 and h <= 130):
		digitCnts.append(c)
		print(len(cnts))


# %%
# sort the contours from left-to-right, then initialize the
# actual digits themselves
digitCnts = contours.sort_contours(digitCnts,
	method="left-to-right")[0]
digits = []


# %%
# loop over each of the digits
for c in digitCnts:
	# extract the digit ROI
	(x, y, w, h) = cv2.boundingRect(c)
	roi = thresh[y:y + h, x:x + w]
	cv2.imshow("Output", roi)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
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
	cv2.rectangle(output, (x, y+offset), (x + w, y + h+offset), (0, 255, 0), 1)
	cv2.putText(output, str(digit), (x - 10, y - 10),
	cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)


number = "{}{}.{}".format(*digits)

cv2.putText(image, str(number), (10, 20),
	cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
print(number)


# %%
# display the digits

# print("{}{}.{}".format(*digits))

cv2.imshow("Input", image)
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()


