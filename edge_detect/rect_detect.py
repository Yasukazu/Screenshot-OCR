from sys import argv, exit
import cv2 
# import numpy as np 
if len(argv) < 2:
    print("Rectangle detector. Needs filespec.")
    exit(1)

# Load the image 
image = cv2.imread(argv[1]) # 'path/to/your/image.jpg') 
 
# Convert to grayscale 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
 
# Apply Gaussian blur to reduce noise and improve edge detection 
blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
 
# Perform edge detection 
edges = cv2.Canny(blurred, 50, 150) 
 
# Find contours in the edged image 
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
 
# Loop over the contours 
for contour in contours: 
    # Approximate the contour to a polygon 
    epsilon = 0.02 * cv2.arcLength(contour, True) 
    approx = cv2.approxPolyDP(contour, epsilon, True) 
     
    # Check if the approximated contour has 4 points (rectangle) 
    if len(approx) == 4: 
        # Draw the rectangle on the original image 
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2) 
 
# Display the result 
cv2.imshow('Detected Rectangles', image) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 