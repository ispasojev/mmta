from collections import Counter
import sys
import cv2 as cv
import numpy as np
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
	# https://medium.com/generalist-dev/background-colour-detection-using-opencv-and-python-22ed8655b243 

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
		# print("Average RGB for top ten is: (", average_red, ", ", average_green, ", ", average_blue, ")")
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


class BackgroundColorDetector_horse():
	# https://medium.com/generalist-dev/background-colour-detection-using-opencv-and-python-22ed8655b243 

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
		print("Average RGB for top ten is: (", average_red, ", ", average_green, ", ", average_blue, ")")
		# if average_blue >225 and average_green >225 and average_red>225:
		# 	print(str(i) + ".jpg IS MATCHED!")
		# 	if i<500 and i>399:
		# 		print("@@@@@ This IS Correct Dinosaur Picture !")
		

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


	

def retrieval():
	print("1: beach")
	print("2: building")
	print("3: bus")
	print("4: dinosaur")
	print("5: flower")
	print("6: horse")
	print("7: man")
	choice = input("Type in the number to choose a category and type enter to confirm\n")
	if choice == '1':
		src_input = cv.imread("beach.jpg")
		print("You choose: %s - beach\n" % choice)
	if choice == '2':
		src_input = cv.imread("building.jpg")
		print("You choose: %s - building\n" % choice)
	if choice == '3':
		src_input = cv.imread("bus.jpg")
		print("You choose: %s - bus\n" % choice)
	if choice == '4':
		src_input = cv.imread("dinosaur.jpg")
		print("You choose: %s - dinosaur\n" % choice)
	if choice == '5':
		src_input = cv.imread("flower.jpg")
		print("You choose: %s - flower\n" % choice)
	if choice == '6':
		src_input = cv.imread("horse.jpg")
		print("You choose: %s - horse\n" % choice)
	if choice == '7':
		src_input = cv.imread("man.jpg")
		print("You choose: %s - man\n" % choice)	

	min_diff = 1e50

	# src_input = cv.imread("man.jpg")

	cv.imshow("Input", src_input)

	# change the image to gray scale
	src_gray = cv.cvtColor(src_input, cv.COLOR_BGR2GRAY)

	# read image database
	database = sorted(glob(database_dir + "/*.jpg"))


	for img in database:
		# read image
		img_rgb = cv.imread(img)
		# convert to gray scale
		img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
		# compare the two images
		diff = compareImgs(src_gray, img_gray)
		# compare the two images by histogram, uncomment the following line to use histogram
		# diff = compareImgs_hist(src_gray, img_gray)
		print(img, diff)
		# find the minimum difference
		if diff <= min_diff:
			# update the minimum difference
			min_diff = diff
			# update the most similar image
			closest_img = img_rgb
			result = img
	
	print("the most similar image is %s, the pixel-by-pixel difference is %f " % (result, min_diff))
	print("\n")

	cv.imshow("Result", closest_img)
	cv.waitKey(0)
	cv.destroyAllWindows()

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


def main():
	# img = cv.imread("beach.jpg")
	# cv.imshow("Image", img)
	# from matplotlib import pyplot as plt
	# plt.hist(img.ravel(),10,[0,256]); plt.show()
	# gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	# cv.imshow("Gray Image", gray_img)
	# cv.waitKey()


	print("1: Image retrieval demo")
	print("2: SIFT demo")
	number = int(input("Type in the number to choose a demo and type enter to confirm\n"))
	if number == 1:
		retrieval()
	elif number == 2:
		# SIFT()
		# pass
		print("1: dinosaur")
		print("2: horse")
		print("3: beach")
		number = int(input("Type in the number to choose a demo and type enter to confirm\n"))
		if number == 1:
			# Presicion = 79/92 = 85.9%
			# recall = 79 / 100 = 79%
			for i in range(1000):
				y= str(i)
				BackgroundColor = BackgroundColorDetector_dinosaur("image.orig/"+y+".jpg")
				# print("Image Number is: "+y+".jpg  ")
				BackgroundColor.detect(i)
		elif number == 2:
			# Presicion = 
			# recall = 
			for i in range(700,800):
				y= str(i)
				BackgroundColor = BackgroundColorDetector_horse("image.orig/"+y+".jpg")
				print("Image Number is: "+y+".jpg  ")
				BackgroundColor.detect(i)
		elif number == 3:
			BackgroundColor = BackgroundColorDetector_dinosaur(sys.argv[0])
			BackgroundColor.detect()
	else:
		print("Invalid input")
		exit()

	# if (len(sys.argv) != 2):
	# 	print("ERROR: syntax is 'python main.py /example/image/location.jpg'")
	# else: 
	# 	BackgroundColor = BackgroundColorDetector(sys.argv[1])
	# 	BackgroundColor.detect()

	# self = cv.imread("image.orig/685.jpg")
	# detect("image.orig/685.jpg")

main()