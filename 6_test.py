import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the palm print image
img = cv2.imread("hand_Scan_H.jpg")

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding
adaptive_thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Apply Non-Local Means Denoising
denoised_img = cv2.fastNlMeansDenoising(adaptive_thresh_img, None, h=10, searchWindowSize=21)

# Display the original and processed images using matplotlib
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Adaptive thresholded image
plt.subplot(1, 3, 2)
plt.title("Adaptive Thresholding")
plt.imshow(adaptive_thresh_img, cmap='gray')
plt.axis('off')

# Denoised image
plt.subplot(1, 3, 3)
plt.title("Denoised Image")
plt.imshow(denoised_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
