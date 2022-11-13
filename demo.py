from collections import Counter
import sys
from tokenize import Double
import cv2 as cv
import pdb
import numpy as np
import math
from tkinter import *
import tkinter as tk
import os
from glob import glob

# the directory of the image database
database_dir = "image.orig"

# Compute pixel-by-pixel difference and return the sum
def compareImgs(img1, img2):
    # resize img2 to img1
	img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))
	diff = cv.absdiff(img1, img2)
	return diff.sum()

def compareImgs_hist(img1, img2):
	width, height = img1.shape[1], img1.shape[0]
	img2 = cv.resize(img2, (width, height))
	num_bins = 10
	hist1 = [0] * num_bins
	hist2 = [0] * num_bins
	bin_width = 255.0 / num_bins + 1e-4
	# compute histogram from scratch

	# for w in range(width):
	# 	for h in range(height):
	# 		hist1[int(img1[h, w] / bin_width)] += 1
	# 		hist2[int(img2[h, w] / bin_width)] += 1

	# compute histogram by using opencv function
	# https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html#ga4b2b5fd75503ff9e6844cc4dcdaed35d

	hist1 = cv.calcHist([img1], [0], None, [num_bins], [0, 255])
	hist2 = cv.calcHist([img2], [0], None, [num_bins], [0, 255])
	sum = 0
	for i in range(num_bins):
		sum += abs(hist1[i] - hist2[i])
	return sum / float(width * height)

def flowerContour(img1, img2):
	kernel = cv.getStructuringElement(cv.MORPH_RECT, (6, 6))
	totalArea1 = 0
	totalArea2 = 0

	# calculation for the area of a circle in img1
	heightCenter1 = img1.shape[0]/2
	widthCenter1 = img1.shape[1]/2
	heightCircle1 = img1.shape[0]/2.3
	widthCircle1 = img1.shape[1]/2.7

	center1 = (round(widthCenter1), round(heightCenter1))
	axesLength1 = (round(widthCircle1), round(heightCircle1))
	circleArea1 = round(math.pi * widthCircle1 * heightCircle1)

    # calculation for contours and total area of img1
	L,U,V = cv.split(cv.cvtColor(img1, cv.COLOR_BGR2LUV))
	channel1 = cv.merge([U, U, U])
	channel1 = cv.cvtColor(channel1, cv.COLOR_BGR2GRAY)
	closed1 = cv.morphologyEx(channel1, cv.MORPH_CLOSE, kernel)
	closed1 = cv.medianBlur(closed1, 3)
	retval, threshold1 = cv.threshold(closed1, 110, 255, cv.THRESH_BINARY)

	contours1, hierarchy1 = cv.findContours(threshold1, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

	for contour in contours1:
		totalArea1 += cv.contourArea(contour)

	# calculation for the area of a circle in img2
	img2copy = img2.copy()

	heightCenter2 = img2copy.shape[0]/2
	widthCenter2 = img2copy.shape[1]/2
	heightCircle2 = img2copy.shape[0]/2.3
	widthCircle2 = img2copy.shape[1]/2.5

	center2 = (round(widthCenter2), round(heightCenter2))
	axesLength2 = (round(widthCircle2), round(heightCircle2))
	circleArea2 = round(math.pi * widthCircle2 * heightCircle2)

	img2circle = cv.ellipse(img2copy, center2, axesLength2, 0, 0, 360, (0, 0, 255), 2)

	# calculation for contours and total area of img2
	L,U,V = cv.split(cv.cvtColor(img2circle, cv.COLOR_BGR2LUV))
	channel2 = cv.merge([U, U, U])
	channel2 = cv.cvtColor(channel2, cv.COLOR_BGR2GRAY)
	closed2 = cv.morphologyEx(channel2, cv.MORPH_CLOSE, kernel)
	closed2 = cv.medianBlur(closed2, 3)
	retval, threshold2 = cv.threshold(closed2, 110, 255, cv.THRESH_BINARY)

	contours2, hierarchy2 = cv.findContours(threshold2, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

	for contour in contours2:
		totalArea2 += cv.contourArea(contour)

	# calculation of contour area over circle area
	circleRatio1 = totalArea1 / circleArea1
	circleRatio2 = totalArea2 / circleArea2
	difference = circleRatio2 / circleRatio1

	return difference

def hsvHist_beach(img1, img2):
	width, height = img1.shape[1], img1.shape[0]
	img2 = cv.resize(img2, (width, height))
	sum = 0
	total_white =0
	total_diff=0
	total_blue=0
	total_sand=0
	total_sand_a =0
	total_sand_b=0
	total_red=0

	hsv1 = cv.cvtColor(img1,cv.COLOR_BGR2HSV)
	hsv2 = cv.cvtColor(img2,cv.COLOR_BGR2HSV)
	h1, s1, v1 = hsv1[:,:,0], hsv1[:,:,1], hsv1[:,:,2]
	h2, s2, v2 = hsv2[:,:,0], hsv2[:,:,1], hsv2[:,:,2]
	hist1 = cv.calcHist([h1], [0], None, [180], [0, 180])
	hist2 = cv.calcHist([h2], [0], None, [180], [0, 180])
	hist21 = cv.calcHist([s1], [0], None, [256], [0, 256])
	hist22 = cv.calcHist([s2], [0], None, [256], [0, 256])

	for i in range(0,10):
		total_white+=hist22[i]
	white_p=total_white/float(width * height)
	if(white_p>0.3):
		total_diff+=20
	for i in range(90,130):
		total_blue+=hist2[i]
	blue_p=total_blue/float(width * height)
	for i in range(15,30):
		total_sand+=hist2[i]
	sand_p=total_sand/float(width * height)
	if(sand_p+blue_p<0.4):
		total_diff+=20
	for i in range(0,5):
		total_red+=hist2[i]
	for i in range(150,180):
		total_red+=hist2[i]
	red_p=total_red/float(width * height)
	if(red_p>0.178):
		total_diff+=20

	# upper segment
	img1a = img1[0:int(height/2), 0:width]
	img2a = img2[0:int(height/2), 0:width]
	hsv1 = cv.cvtColor(img1a,cv.COLOR_BGR2HSV)
	hsv2 = cv.cvtColor(img2a,cv.COLOR_BGR2HSV)
	h1, s1, v1 = hsv1[:,:,0], hsv1[:,:,1], hsv1[:,:,2]
	h2, s2, v2 = hsv2[:,:,0], hsv2[:,:,1], hsv2[:,:,2]
	hist1 = cv.calcHist([h1], [0], None, [180], [0, 180])
	hist2 = cv.calcHist([h2], [0], None, [180], [0, 180])
	hist21 = cv.calcHist([s1], [0], None, [256], [0, 256])
	hist22 = cv.calcHist([s2], [0], None, [256], [0, 256])
	# h channel
	for i in range(180):
		sum += abs(hist1[i] - hist2[i])
	# s channel
	for i in range(256):
		sum += abs(hist21[i] - hist22[i])

	for i in range(15,30):
		total_sand_a+=hist2[i]
	sand_ap=total_sand_a/float(width * height)

	# lower segment
	img1b = img1[int(height/2):height, 0:width]
	img2b = img2[int(height/2):height, 0:width]
	hsv1 = cv.cvtColor(img1b,cv.COLOR_BGR2HSV)
	hsv2 = cv.cvtColor(img2b,cv.COLOR_BGR2HSV)
	h1, s1, v1 = hsv1[:,:,0], hsv1[:,:,1], hsv1[:,:,2]
	h2, s2, v2 = hsv2[:,:,0], hsv2[:,:,1], hsv2[:,:,2]
	hist1 = cv.calcHist([h1], [0], None, [180], [0, 180])
	hist2 = cv.calcHist([h2], [0], None, [180], [0, 180])
	hist21 = cv.calcHist([s1], [0], None, [256], [0, 256])
	hist22 = cv.calcHist([s2], [0], None, [256], [0, 256])
	# h channel
	for i in range(180):
		sum += abs(hist1[i] - hist2[i])
	# s channel
	for i in range(256):
		sum += abs(hist21[i] - hist22[i])

	for i in range(15,30):
		total_sand_b+=hist2[i]
	sand_bp=total_sand_b/float(width * height)

	if(sand_ap>sand_bp+0.05):
		total_diff+=20

	diff1 = compareImgs(img1, img2) * 3 / 10000000

	total_diff += diff1
	total_diff/=100
	return total_diff

def hsvHist_horse(img1, img2):
	width, height = img1.shape[1], img1.shape[0]
	img2 = cv.resize(img2, (width, height))
	sum = 0
	total_green =0
	total_diff=0

	hsv1 = cv.cvtColor(img1,cv.COLOR_BGR2HSV)
	hsv2 = cv.cvtColor(img2,cv.COLOR_BGR2HSV)
	h1, s1, v1 = hsv1[:,:,0], hsv1[:,:,1], hsv1[:,:,2]
	h2, s2, v2 = hsv2[:,:,0], hsv2[:,:,1], hsv2[:,:,2]

	hist1 = cv.calcHist([h1], [0], None, [180], [0, 180])
	hist2 = cv.calcHist([h2], [0], None, [180], [0, 180])
	hist21 = cv.calcHist([s1], [0], None, [256], [0, 256])
	hist22 = cv.calcHist([s2], [0], None, [256], [0, 256])

	for i in range(35,75):
		total_green+=hist2[i]
	green_p=total_green/float(width * height)
	if(green_p<0.45):
		add_index=20
	else:
		add_index=0

	# h channel
	for i in range(180):
		sum += abs(hist1[i] - hist2[i])
	# s channel
	for i in range(256):
		sum += abs(hist21[i] - hist22[i])
	total_diff=(sum / float(width * height)) * 20 + add_index
	total_diff/=100
	return (total_diff)

def checkMaxDifference(diffs):
    maxValIdx = 0
    maxVal = diffs[maxValIdx]
    i = 0
    for current in diffs:
        # if current bigger than/equals maxVal -> replace
        if current >= maxVal:
            maxVal = current
            maxValIdx = i
        i +=1
    return maxValIdx, maxVal

def checkMinDifference(diffs):
    minValIdx = 0
    minVal = diffs[minValIdx]
    i = 0
    for current in diffs:
        # if current smaller than/equals minVal -> replace
        if current <= minVal:
            minVal = current
            minValIdx = i
        i +=1
    return minValIdx, minVal

def compute_SIFTdiff(img1, img2):
	#-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
	minHessian = 400
	detector = cv.SIFT_create()
	#detector = cv.xfeatures2d.BEBLID_create(0.75)
	keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
	keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
	#-- Step 2: Matching descriptor vectors with a brute force matcher
	matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)
	#matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
	#matcher = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
	matches = matcher.match(descriptors1, descriptors2)

	matches = sorted(matches, key = lambda x:x.distance)
	min_dist = matches[0].distance
	good_matches = tuple(filter(lambda x:x.distance <= 1.2 * min_dist, matches)) #TODO: parameter adaptable
	return len(good_matches)

def compute_ORBdiff(img1, img2):
	detector = cv.ORB_create(nfeatures=500) #ORB better for flowers with 1.2*min_dist and blurred src_input, n_features=500
	keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
	keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

	matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)
	matches = matcher.match(descriptors1, descriptors2)

	matches = sorted(matches, key = lambda x:x.distance)
	min_dist = matches[0].distance
	good_matches = tuple(filter(lambda x:x.distance <= 1.2 * min_dist, matches))
	return len(good_matches)

def SIFT():
	img1 = cv.imread("flower.jpg")
	img2 = cv.imread("image.orig/685.jpg")
	if img1 is None or img2 is None:
		print('Error loading images!')
		exit(0)
	#-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
	minHessian = 400
	detector = cv.SIFT_create()
	keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
	keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
	#-- Step 2: Matching descriptor vectors with a brute force matcher
	matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)
	matches = matcher.match(descriptors1, descriptors2)
	#-- Draw matches
	img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
	cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches)
	#-- Show detected matches
	cv.imshow('Matches: SIFT (Python)', img_matches)
	cv.waitKey()

	# draw good matches
	matches = sorted(matches, key = lambda x:x.distance)
	min_dist = matches[0].distance
	good_matches = tuple(filter(lambda x:x.distance <= 2 * min_dist, matches))

	img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
	cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	#-- Show detected matches
	cv.imshow('Good Matches: SIFT (Python)', img_matches)
	cv.waitKey()

def retrieve_images(choice):
	print("1: beach")
	print("2: building")
	print("3: bus")
	print("4: dinosaur")
	print("5: flower")
	print("6: horse")
	print("7: man")
	if choice == '1':
		category_name = "beach"
		chosenCategory = 2
		threshold = 0.5
		src_input = cv.imread("beach.jpg")
		print("You choose: %s - beach\n" % choice)
	if choice == '2':
		category_name = "building"
		chosenCategory = 3
		threshold = 0.5
		src_input = cv.imread("building.jpg")
		print("You choose: %s - building\n" % choice)
	if choice == '3':
		category_name = "bus"
		chosenCategory = 4
		threshold = 0.09
		src_input = cv.imread("bus.jpg")
		print("You choose: %s - bus\n" % choice)
	if choice == '4':
		chosenCategory = 5
		threshold = 0.5
		category_name = "dinosaur"
		src_input = cv.imread("dinosaur.jpg")
		print("You choose: %s - dinosaur\n" % choice)
	if choice == '5':
		chosenCategory = 7
		threshold = 0.92
		category_name = "flower"
		src_input = cv.imread("flower.jpg")
		print("You choose: %s - flower\n" % choice)
	if choice == '6':
		chosenCategory = 8
		threshold = 0.61
		category_name = "horse"
		src_input = cv.imread("horse.jpg")
		print("You choose: %s - horse\n" % choice)
	if choice == '7':
		category_name = "man"
		chosenCategory = 1
		threshold = 0.45
		src_input = cv.imread("man.jpg")
		print("You choose: %s - man\n" % choice)

	min_diff = 1e50

	# src_input = cv.imread("man.jpg")

	cv.imshow("Input", src_input)

	# change the image to gray scale
	src_gray = cv.cvtColor(src_input, cv.COLOR_BGR2GRAY)

	# read image database
	database = sorted(glob(database_dir + "/*.jpg"), key = len)

	# initialize arrays for fixed size of retrieval_amount
	min_diffs = []
	closest_imgs = []
	result = []

	diff = 0

	if choice == '7': # choice is human, we need face detector
		faces_amount = []
		id_img_w_faces = []

		diffSum = 0
		minDiff_thres = 999999999
		maxDiff_thres = 0

		for img in database:
			img_rgb = cv.imread(img)
			img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

			# face detection
			face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
			faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)
			if len(faces)>0:
				faces_amount.append(len(faces)) # how many faces found in img
				id_img_w_faces.append(img) # save img name

		for img in id_img_w_faces:
			img_rgb = cv.imread(img)
			diff = compareImgs_hist(src_input, img_rgb)

			# Necessary parameteres to compute threshold
			diffSum += diff
			if diff <= minDiff_thres:
				minDiff_thres = diff
			if diff >= maxDiff_thres:
				maxDiff_thres = diff

		thresValue = computeAllowedDiff(diffSum, minDiff_thres, maxDiff_thres, threshold)
		for img in id_img_w_faces:
			img_rgb = cv.imread(img)
			diff = compareImgs_hist(src_input, img_rgb)

			if diff <= thresValue:
				min_diffs.append(diff)
				result.append(img)
				closest_imgs.append(img_rgb)

	else: # choice is not human, we use other algorithms
		diffSum = 0
		minDiff_thres = 999999999
		maxDiff_thres = 0
		for img in database:
			img_rgb = cv.imread(img)
			img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

			# compare the two images
			if choice == '1': #beach
				diff = diff = hsvHist_beach(src_input, img_rgb)

			elif choice == '2': #building
				diff = compareImgs(src_gray, img_gray)

			elif choice == '3': #bus
				diff = compareImgs_hist(src_gray, img_gray)

			elif choice == '4': # dinosaur
				diff = compareImgs(src_gray, img_gray)

			elif choice == '5': #flower
				diff = flowerContour(src_input, img_rgb)

			elif choice == '6': #horse
				diff = hsvHist_horse(src_input, img_rgb)

			print(img, diff)

			diffSum += diff
			if diff <= minDiff_thres:
				minDiff_thres = diff
			elif diff >= maxDiff_thres:
				maxDiff_thres = diff

		thresValue = computeAllowedDiff(diffSum, minDiff_thres, maxDiff_thres, threshold)
		print("thresValue", thresValue)

		for img in database:
			img_rgb = cv.imread(img)
			img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

			if choice == '1': #beach
				diff = diff = hsvHist_beach(src_input, img_rgb)

			elif choice == '2': #building
				diff = compareImgs(src_gray, img_gray)

			elif choice == '3': # bus
				diff = compareImgs_hist(src_gray, img_gray)

			elif choice == '4': # dinosaur
				diff = compareImgs(src_gray, img_gray)

			elif choice == '5': #flower
				diff = flowerContour(src_input, img_rgb)

			elif choice == '6': # horse
				diff = hsvHist_horse(src_input, img_rgb)

			# check if current difference <= threshold and append to result if so.
			if choice == '1' or choice == '2' or choice == '3' or choice == '4' or choice == '5' or choice == '6':
				if diff <= thresValue:
					min_diffs.append(diff)
					result.append(img)
					closest_imgs.append(img_rgb)

    # initializing list of retrieved images
	retrieved_images = []

	# formula to take multiple images
	j=0
	img_id = 0
	for img in closest_imgs:
		print("the most similar images are %s, the pixel-by-pixel difference is %f " % (result[j], min_diffs[j]))
		if len(result[j]) == 18:
			img_id = int(result[j][11:14])
		if len(result[j]) == 17:
			img_id = int(result[j][11:13])
		if len(result[j]) == 16:
			img_id = int(result[j][11:12])
		cv.imshow("Result " + str(j), closest_imgs[j])
		retrieved_images.append(img_id)
		j+=1

    # calculation of the recall and precision rate
	inCategory = 0
	for i in retrieved_images:
		if i in range((chosenCategory * 100) - 100, (chosenCategory * 100) - 1):
			inCategory += 1
	recall_rate = inCategory * 1.0
	if len(result)>0:
		precision_rate = inCategory / len(result) * 100.0
	else:
		precision_rate = 0

	for img_name in result:
		if len(img_name) == 18:
			img_num = img_name[11:18]
		if len(img_name) == 17:
			img_num = img_name[11:17]
		if len(img_name) == 16:
			img_num = img_name[11:16]
		img_path = "./result/" + category_name + "/" + img_num
		img = cv.imread(img_name)
		isWritten = cv.imwrite(img_path, img)
		if isWritten:
			print('Image ', img_name ,' is successfully saved.')

	# print the recall and precision rate
	print("\n")
	print("Recall Rate: " + str(recall_rate) + "%")
	print("Precision Rate: " + str(precision_rate) + "%")

	return category_name, precision_rate, recall_rate

def computeAllowedDiff(diffSum, minDiff, maxDiff, threshold):
    mean = diffSum/1000
    diffRange = maxDiff - minDiff
    return (diffRange * threshold)

def main():

	# initialize the gui
	root = Tk()
	root.geometry("1280x720")
	root.configure(bg="#e3fffd")
	root.title("CS4185 Image Retrieval")

	# run the image retrieval through the gui
	def retrieveImage():
		input = str(category.get())
		
		# show the precision and recall rate in the gui
		category_name, precisionrate, recallrate = retrieve_images(input)

		categoryDisplay.config(text="Category = " + category_name)
		precisionDisplay.config(text="Precision Rate = " + str(precisionrate) + "%")
		recallDisplay.config(text="Recall Rate = " + str(recallrate) + "%")

	# show the categories in the gui
	categories = [
		("Beach", 1),
		("Building", 2),
		("Bus", 3),
		("Dinosaur", 4),
		("Flower", 5),
		("Horse", 6),
		("Man", 7)
	]

	category = IntVar()
	category.set(0)

	categoryDisplay = Label(root, text="Category = ", font=("Calibri", 15), bg="#e3fffd")

	precisionDisplay = Label(root, text="Precision Rate = % ", font=("Calibri", 15), bg="#e3fffd")

	recallDisplay = Label(root, text="Recall Rate = % ", font=("Calibri", 15), bg="#e3fffd")

	exitButtonDisplay = Button(root, text="Exit Program", font=("Calibri", 15), command=root.destroy)

	for text, categoryName in categories:
		Radiobutton(root, text=text, variable=category, value=categoryName, command=lambda: [retrieveImage()], bg="#e3fffd", 
		activebackground="#ffd3bf", font=("Calibri", 15)).pack(anchor=W)

	categoryDisplay.pack(anchor=W)
	precisionDisplay.pack(anchor=W)
	recallDisplay.pack(anchor=W)
	exitButtonDisplay.pack(pady=20)

	mainloop()

main()
