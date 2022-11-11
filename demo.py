from collections import Counter
import sys
from tokenize import Double
import cv2 as cv
import pdb
import numpy as np
import math
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

class BackgroundColorDetector_dinosaur():

	def __init__(self, imageLoc):
		self.img = cv.imread(imageLoc, 1)
		self.manual_count = {}
		self.w, self.h, self.channels = self.img.shape
		self.total_pixels = self.w*self.h

	def count(self):
		for y in range(0, self.h):
			for x in range(0, self.w):
				RGB = (self.img[x, y, 2], self.img[x, y, 1], self.img[x, y, 0])
				if RGB in self.manual_count:
					self.manual_count[RGB] += 1
				else:
					self.manual_count[RGB] = 1

	def average_colour(self,i):
		red = 0
		green = 0
		blue = 0
		sample = 10
		for top in range(0, sample):
			red += self.number_counter[top][0][0]
			green += self.number_counter[top][0][1]
			blue += self.number_counter[top][0][2]

		average_red = red / sample
		average_green = green / sample
		average_blue = blue / sample
		# print("Average RGB for top twenty is: (", average_red, ", ", average_green, ", ", average_blue, ")")
		if average_blue >225 and average_green >225 and average_red>225:
			print(str(i) + ".jpg IS MATCHED!")
			if i<500 and i>399:
				print("@@@@@ This IS Correct Dinosaur Picture !")


	def twenty_most_common(self):
		self.count()
		self.number_counter = Counter(self.manual_count).most_common(20) #### 20->10
		# for rgb, value in self.number_counter:
		# 	print(rgb, value, ((float(value)/self.total_pixels)*100))

	def detect(self,i):
		self.twenty_most_common()
		self.percentage_of_first = (
			float(self.number_counter[0][1])/self.total_pixels)
		# print(self.percentage_of_first)
		if self.percentage_of_first > 0.5:
			print("Background color is ", self.number_counter[0][0])

		else:
			self.average_colour(i)

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

def busContour(img1, img2):
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

def retrieval(retrieval_amount):
	print("1: beach")
	print("2: building")
	print("3: bus")
	print("4: dinosaur")
	print("5: flower")
	print("6: horse")
	print("7: man")
	choice = input("Type in the number to choose a category and type enter to confirm\n")
	if choice == '1':
		chosenCategory = 2
		src_input = cv.imread("beach.jpg")
		print("You choose: %s - beach\n" % choice)
	if choice == '2':
		chosenCategory = 3
		src_input = cv.imread("building.jpg")
		print("You choose: %s - building\n" % choice)
	if choice == '3':
		chosenCategory = 4
		src_input = cv.imread("bus.jpg")
		print("You choose: %s - bus\n" % choice)
	if choice == '4':
		chosenCategory = 5
		src_input = cv.imread("dinosaur.jpg")
		print("You choose: %s - dinosaur\n" % choice)
		# for i in range(1000):
		# 		y= str(i)
		# 		BackgroundColor = BackgroundColorDetector_dinosaur("image.orig/"+y+".jpg")
		# 		# print("Image Number is: "+y+".jpg  ")
		# 		BackgroundColor.detect(i)
	if choice == '5':
		chosenCategory = 7
		src_input = cv.imread("flower.jpg")
		print("You choose: %s - flower\n" % choice)
	if choice == '6':
		chosenCategory = 8
		src_input = cv.imread("horse.jpg")
		print("You choose: %s - horse\n" % choice)
	if choice == '7':
		chosenCategory = 1
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
	min_diffs = [999999999.0] * retrieval_amount
	closest_imgs = [0] * retrieval_amount
	result = [0] * retrieval_amount
	# initialize maxVal
	maxValIdx, maxVal = checkMaxDifference(min_diffs)
 
	# for SIFT/ORB, we need max_diffs and minVal
	max_diffs = [0] * retrieval_amount
	minValIdx, minVal = checkMinDifference(max_diffs)
	diff = 0
 
	if choice == '7':
		faces_amount = []
		id_img_w_faces = []		
		minFaces = 0
		minFacesIdx = 0
		i = 0

		for img in database:
			img_rgb = cv.imread(img)
			img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

			# face detection
			face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
			faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)
			if len(faces)>0:
				faces_amount.append(len(faces)) # how many faces found in img
				id_img_w_faces.append(img) # save img name
				if len(faces)<= minFaces:
					minFacesIdx = i # same img idx for smallest amount of found faces in img 
					#TODO - optional: Enhance performance of recall/precision by includig all with more than 1 face detected 
     
		for img in id_img_w_faces:
			img_rgb = cv.imread(img)
			diff = compareImgs_hist(src_input, img_rgb)

			if diff <= maxVal:
					# update the minimum difference
					min_diffs[maxValIdx] = diff
					# update the most similar image
					closest_imgs[maxValIdx] = img_rgb
					result[maxValIdx] = img
					# update max difference in min_diffs array
					maxValIdx, maxVal = checkMaxDifference(min_diffs)
   
	else: # choice is not human, we use other algorithms
		for img in database:
			# read image
			img_rgb = cv.imread(img)
			# convert to gray scale
			img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
			
			# compare the two images
			if choice == '1' or choice == '2': #beach, building
				diff = compareImgs(src_gray, img_gray)

			elif choice == '3': #bus
				#dst = cv.GaussianBlur(src_input,(25,25),0)
				#diff = compute_ORBdiff(dst, img_rgb)
				diff = compute_SIFTdiff(src_input, img_rgb)
				if diff >= minVal:
					max_diffs[minValIdx] = diff
					closest_imgs[minValIdx] = img_rgb
					result[minValIdx] = img
					minValIdx, minVal = checkMinDifference(max_diffs)

			elif choice == '4': # dinosaur
				diff = compareImgs(src_gray, img_gray)

			elif choice == '5': #flower
				diff = flowerContour(src_input, img_rgb)

			elif choice == '6': #horse
				diff = compareImgs(src_input, img_rgb)

			print(img, diff)
			# find the minimum difference
			if diff <= maxVal:
				# update the minimum difference
				min_diffs[maxValIdx] = diff
				# update the most similar image
				closest_imgs[maxValIdx] = img_rgb
				result[maxValIdx] = img
				# update max difference in min_diffs array
				maxValIdx, maxVal = checkMaxDifference(min_diffs)

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
	precision_rate = inCategory / retrieval_amount * 100.0
	
	# print the recall and precision rate
	print("\n")
	print("Recall Rate: " + str(recall_rate) + "%")
	print("Precision Rate: " + str(precision_rate) + "%")

	cv.waitKey(0)
	cv.destroyAllWindows()
 
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
 
def retrieve_threshold(threshold):
	print("1: beach")
	print("2: building")
	print("3: bus")
	print("4: dinosaur")
	print("5: flower")
	print("6: horse")
	print("7: man")
	choice = input("Type in the number to choose a category and type enter to confirm\n")
	if choice == '1':
		category_name = "beach"
		chosenCategory = 2
		src_input = cv.imread("beach.jpg")
		print("You choose: %s - beach\n" % choice)
	if choice == '2':
		category_name = "building"
		chosenCategory = 3
		src_input = cv.imread("building.jpg")
		print("You choose: %s - building\n" % choice)
	if choice == '3':
		category_name = "bus"
		chosenCategory = 4
		src_input = cv.imread("bus.jpg")
		print("You choose: %s - bus\n" % choice)
	if choice == '4':
		chosenCategory = 5
		category_name = "dinosaur"
		src_input = cv.imread("dinosaur.jpg")
		print("You choose: %s - dinosaur\n" % choice)
		# for i in range(1000):
		# 		y= str(i)
		# 		BackgroundColor = BackgroundColorDetector_dinosaur("image.orig/"+y+".jpg")
		# 		# print("Image Number is: "+y+".jpg  ")
		# 		BackgroundColor.detect(i)
	if choice == '5':
		chosenCategory = 7
		category_name = "flower"
		src_input = cv.imread("flower.jpg")
		print("You choose: %s - flower\n" % choice)
	if choice == '6':
		chosenCategory = 8
		category_name = "horse"
		src_input = cv.imread("horse.jpg")
		print("You choose: %s - horse\n" % choice)
	if choice == '7':
		category_name = "man"
		chosenCategory = 1
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
	# initialize maxVal
	#maxValIdx, maxVal = checkMaxDifference(min_diffs)
 
	# for SIFT/ORB, we need max_diffs and minVal
	max_diffs = []
	#minValIdx, minVal = checkMinDifference(max_diffs)
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
			if choice == '1' or choice == '2': # beach, building
				diff = compareImgs(src_gray, img_gray)

			elif choice == '3': #bus
				diff = compute_SIFTdiff(src_input, img_rgb)

			elif choice == '4': # dinosaur
				diff = compareImgs(src_gray, img_gray)

			elif choice == '5': #flower
				diff = flowerContour(src_input, img_rgb)

			elif choice == '6': #horse
				diff = compareImgs(src_input, img_rgb)

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

			if choice == '1' or choice == '2':
				diff = compareImgs(src_gray, img_gray)
			
			elif choice == '3': # bus
				diff = compute_SIFTdiff(src_input, img_rgb)
				if diff >= minVal:
					max_diffs[minValIdx] = diff
					closest_imgs[minValIdx] = img_rgb
					result[minValIdx] = img
					minValIdx, minVal = checkMinDifference(max_diffs)

			elif choice == '4': # dinosaur
				diff = compareImgs(src_gray, img_gray)

			elif choice == '5': #flower
				diff = flowerContour(src_input, img_rgb)

			elif choice == '6': # horse
				diff = compareImgs(src_input, img_rgb)

			# check if current difference <= threshold and append to result if so.
			if choice == '1' or choice == '2' or choice == '4' or choice == '5' or choice == '6':
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

	cv.waitKey(0)
	cv.destroyAllWindows()
 
def computeAllowedDiff(diffSum, minDiff, maxDiff, threshold):
    mean = diffSum/1000
    diffRange = maxDiff - minDiff
    return (diffRange * threshold)
	
def main():

	print("1: Retrieve certain amount of images")
	print("2: Retrieve all images above threshold")
	number = int(input("Type in the number to choose a demo and type enter to confirm\n"))
	if number == 1:
		print("How many images do you want to retrieve?")
		numRetrievedImg = int(input(""))
		if numRetrievedImg > 0:
			retrieval(numRetrievedImg)
		else:
			print("Invalid input")
			exit()
	elif number == 2:
		print("Input threshold from 0 to 1, e.g. 0.5 (0=loose, 1=strict)")
		threshold = float(input(""))
		print("threshold", threshold)
		print("threshold type", type(threshold))
		if 0 <= threshold <= 1:
			retrieve_threshold(threshold)
		else:
			print("Invalid input")
			exit()
	else:
		print("Invalid input")
		exit()

main()