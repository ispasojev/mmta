import cv2 as cv
import numpy as np
from glob import glob

# the directory of the image database
database_dir = "image.orig"
numRetrievedImg = 10

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
 
	# initialize arrays for fixed size of numRetrievedImg
	min_diffs = [999999999.0] * numRetrievedImg
	closest_imgs = [0] * numRetrievedImg
	result = [0] * numRetrievedImg
	# initialize maxVal
	maxValIdx, maxVal = checkMaxDifference(min_diffs)
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
		if diff <= maxVal:
			# update the minimum difference
			min_diffs[maxValIdx] = diff
			# update the most similar image
			closest_imgs[maxValIdx] = img_rgb
			result[maxValIdx] = img
			# update max difference in min_diffs array
			maxValIdx, maxVal = checkMaxDifference(min_diffs)

	j=0
	for img in closest_imgs:
		print("the most similar image is %s, the pixel-by-pixel difference is %f " % (result[j], min_diffs[j]))
		cv.imshow("Result " + str(j), closest_imgs[j])
		j+=1
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
		SIFT()
		# pass
	else:
		print("Invalid input")
		exit()

main()