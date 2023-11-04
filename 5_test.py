import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the palm print image
img = cv2.imread("hand_Scan_H.jpg")

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding
adaptive_thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Adjust Contrast and Brightness
alpha = 1.5  # Contrast control (1.0 means no change)
beta = 10  # Brightness control (0 means no change)
adjusted_img = cv2.convertScaleAbs(adaptive_thresh_img, alpha=alpha, beta=beta)

# Enhance Edges (Unsharp Masking)
blurred_img = cv2.GaussianBlur(adjusted_img, (5,5), 0)
sharp_img = cv2.addWeighted(adjusted_img, 1.5, blurred_img, -0.5, 0)

# Apply CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = clahe.apply(sharp_img)  # Applying CLAHE after enhancing edges

# Display the original image
plt.figure(figsize=(6, 6))
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Display the adaptive thresholded image
plt.figure(figsize=(6, 6))
plt.title("Adaptive Thresholding")
plt.imshow(adaptive_thresh_img, cmap='gray')
plt.axis('off')
plt.show()

# Display the adjusted image with contrast and brightness
plt.figure(figsize=(6, 6))
plt.title("Adjusted Image")
plt.imshow(adjusted_img, cmap='gray')
plt.axis('off')
plt.show()

# Display the CLAHE image after edge enhancement
plt.figure(figsize=(6, 6))
plt.title("CLAHE (with Edge Enhancement)")
plt.imshow(clahe_img, cmap='gray')
plt.axis('off')
plt.show()
