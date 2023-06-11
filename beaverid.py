#####
# 11 June 2023
# Individual recognition of Eurasian beavers (Castor fiber) by their tail pattern using a computer-assisted pattern-identification algorithm.
#####

import os
import random
from shutil import copyfile
from os import listdir
import cv2
import numpy as np
from matplotlib import pyplot

path = #Input where the data needs to be examined

directory_contents = os.listdir(path)
beaver_id = []

for n in range(0,len(directory_contents)):
    beaver_id.append(directory_contents[n])

# Sorting list of Integers in ascending
beaver_id.sort(key=int)

####
# Split data set and test data base
####

# create subdirectories
subdirs = ['data_base/', 'test/']
data_path = path.split("/")
print(data_path)
for subdir in subdirs:
    # create label subdirectories
    for labldir in beaver_id:
        newdir = os.path.join(data_path[0],data_path[1], subdir,labldir)
        os.makedirs(newdir, exist_ok=True)

# seed random number generator
random.seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.70
# copy training dataset images into subdirectories
src_directory = path
for i, file in enumerate(listdir(src_directory)):
    src = src_directory + '/' + file

    for subfile in listdir(src):
        dst_dir = 'data_base/'
        if random.random() < val_ratio:
            dst_dir = 'test/'
    
        dst = os.path.join(data_path[0],data_path[1], dst_dir, file, subfile)
        copyfile(os.path.join(src, subfile), dst)

#####
# Prediction using SIFT algorithm
#####

#####
# Single individual image matching
#####
# Iterate all the folder inside the data base

y_pred = []
y_true = []

id_prediction = []
match_lines = []
image_name = []

for labldir in beaver_id:
    src = os.path.join(data_path[0],data_path[1], 'data_base',str(labldir))

    for file_name in listdir(src):
        print(os.path.join(src, file_name))

        fingerprint_database_image = cv2.imread(os.path.join("/Users/gagad/Desktop/beaver/",src, file_name)) # Input the main path
        fingerprint_database_image = cv2.resize(fingerprint_database_image,(640,480))
        sift = cv2.xfeatures2d.SIFT_create()

        # load image pixels
        test_image = cv2.imread("/Users/gagad/Desktop/beaver/sources/sample_1/original/1/19.jpg") # Input a single image of the beaver tail
        test_image = cv2.resize(test_image,(640,480))

        keypoints_1, descriptors_1 = sift.detectAndCompute(test_image, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_database_image, None)


        matches = cv2.FlannBasedMatcher(dict(algorithm=2, trees=10), 
                dict()).knnMatch(descriptors_1, descriptors_2, k=2)

        match_points = []
        
        for p, q in matches:
            if p.distance < 0.75*q.distance:
                match_points.append(p)

        keypoints = 0
        if len(keypoints_1) <= len(keypoints_2):
            keypoints = len(keypoints_1)            
        else:
            keypoints = len(keypoints_2)

        print(len(match_points))
        print("% match: ", len(match_points) / keypoints * 100)
        print("Figerprint ID: " + str(file_name)) 

        id_prediction = np.append(id_prediction,int(labldir))
        match_lines = np.append(match_lines, len(match_points))
        image_name = np.append(image_name,file_name)


sorted_rank = sorted(zip(match_lines, id_prediction, image_name), reverse=True)[:3]
y_pred = np.append(y_pred, int(id_prediction[0]))

print(sorted_rank)
print("y_pred : ",y_pred)