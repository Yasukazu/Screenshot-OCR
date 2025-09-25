from math import sqrt, floor, dist
from typing import Sequence
from sys import argv, exit
import cv2 
import numpy as np 
from numpy import ndarray

def main(filename: str, cutoff=0.3):
    # Load the image 
    image = cv2.imread(filename) # 'path/to/your/image.jpg') 
    image_h, image_w, _ = image.shape 
    image_min = min([image_h, image_w])
    # Convert to grayscale 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    # Apply Gaussian blur to reduce noise and improve edge detection 
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
    
    # Perform edge detection 
    edges = cv2.Canny(blurred, 50, 150) 
    
    # Find contours in the edged image 
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    
    image_cutoff = floor(image_min * cutoff) # / 16) or 4
    # Loop over the contours 
    for contour in contours: 
        # Approximate the contour to a polygon 
        epsilon = 0.02 * cv2.arcLength(contour, True) 
        approx_cont = cv2.approxPolyDP(contour, epsilon, True) 
        
        # Check if the approximated contour has 4 points (rectangle) 
        if len(approx_cont) == 4:
            min_dist = get_min_distance(approx_cont)
            if min_dist > image_cutoff: 
                # Draw the rectangle on the original image 
                cv2.drawContours(image, [approx_cont], -1, (0, 255, 0), 2) 
    
    # Display the result 
    cv2.imshow('Detected Rectangles', image) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

def get_min_distance(cont: ndarray) -> int: # Sequence[Sequence[int]]) -> int:
    dist_list = []
    for i in range(4):
        p = cont[i, 0] # - cont[i, 0, 0]
        q = cont[(i + 1) % 4, 0] # - cont[i, 0, 0]
        d = floor(dist(p, q))
        if not d:
            return 0
        dist_list.append(d)
        # if d == 0: return 0
        # dw = cont[(i + 1) % 4, 0, 1] - cont[i, 0, 1]
        # dhdw = dh^2 + dw^2
    return min(dist_list)

if __name__ == '__main__':
    if len(argv) < 2:
        print("Rectangle detector. Needs filespec.")
        exit(1)
    main(argv[1], cutoff=float(argv[2]))