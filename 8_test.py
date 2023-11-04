import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhance_palm_print(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    adaptive_thresh_img = cv2.adaptiveThreshold(gray_img, 127, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply Unsharp Masking for auto-sharpness control
    blurred_img = cv2.GaussianBlur(adaptive_thresh_img, (5, 5), 0)

    a = .18 # contrast = 5. Contrast control ( 0 to 127)
    b = 55   # brightness = 2. Brightness control (0-100)
    c = 0
    sharp_img = cv2.addWeighted(adaptive_thresh_img, a, blurred_img, c, b)

    # Apply Non-Local Means Denoising
    denoised_img = cv2.fastNlMeansDenoising(sharp_img, None, h=3, searchWindowSize=17)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(denoised_img)

    return enhanced_img

# Example usage:
enhanced_image = enhance_palm_print("hand_Scan_H.jpg")

# Display the enhanced image
plt.figure(figsize=(8, 8))
plt.title("Enhanced Image")
plt.imshow(enhanced_image, cmap='gray')
plt.axis('off')
plt.show()