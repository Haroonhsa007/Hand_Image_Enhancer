"""
    algo for hand detection from image
"""

import cv2
import numpy as np

# Load the image
image = cv2.imread('hand_Scan_H.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a skin color filter
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)
skin_mask = cv2.inRange(image, lower_skin, upper_skin)

# Apply a series of morphological operations to clean up the mask
kernel = np.ones((5,5),np.uint8)
skin_mask = cv2.erode(skin_mask,kernel,iterations = 1)
skin_mask = cv2.dilate(skin_mask,kernel,iterations = 1)

# Find contours in the skin mask
contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours (noise)
min_contour_area = 1000
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# Draw contours on the original image
cv2.drawContours(image, filtered_contours, -1, (0,255,0), 3)

# Display the result
print(filtered_contours)

cv2.imshow('Hand Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
