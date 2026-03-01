#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 10:30:46 2023

@author: janko
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

video = cv2.VideoCapture("slow_traffic_small.mp4")

# ShiTomasi corner detection
config_st = {'maxCorners': 100,
             'qualityLevel': 0.3,
             'minDistance': 7,
             'blockSize': 7}

# Lucas-Kanade optical flow
config_lk = {'winSize': (15, 15),
             'maxLevel': 2,
             'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find keypoints
ret, source = video.read()
assert ret

src_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
p_src = cv2.goodFeaturesToTrack(src_gray, mask=None, **config_st)

# Create a mask image for drawing purposes
mask = np.zeros_like(source)

while True:
    ret, target = video.read()
    if not ret:
        print('End of video.')
        break
    
    # Convert frame to gray and calculate LK optical flow
    dst_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    p_dst, status, err = cv2.calcOpticalFlowPyrLK(src_gray, dst_gray, p_src, None, **config_lk)
    
    # Select points that have been successfully tracked    
    if p_dst is not None:
        p_dst = p_dst[status==1]
        p_src = p_src[status==1]    
        
    # Draw the tracks
    for i, (dst, src) in enumerate(zip(p_dst, p_src)):
        x_dst, y_dst = dst
        x_src, y_src = src
        
        mask = cv2.line(mask, (int(x_src), int(y_src)), (int(x_dst), int(y_dst)), color[i].tolist(), 2)    
        target = cv2.circle(target, (int(x_src), int(y_src)), 5, color[i].tolist(), -1)

    result = cv2.add(target, mask)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.draw(), plt.show()
    plt.waitforbuttonpress(1/25)
    plt.clf()
    
    # Update the previous frame and previous points
    src_gray = np.copy(dst_gray)
    p_src = p_dst.reshape(-1, 1, 2)
        
