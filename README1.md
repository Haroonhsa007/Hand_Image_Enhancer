```
# Palm Print Enhancement

This Python code is used to enhance a palm print image. It applies various image processing techniques to improve the visibility and quality of the palm print.

## Code Explanation

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```
- This code imports the necessary libraries:
  - `cv2`: OpenCV library for computer vision tasks.
  - `numpy`: Library for numerical computations.
  - `matplotlib.pyplot`: Library for creating visualizations.

```python
def enhance_palm_print(image_path):
```
- This defines a function named `enhance_palm_print` that takes the file path of the palm print image as input.

```python
    # Load the image
    img = cv2.imread(image_path)
```
- It loads the input image from the provided file path using OpenCV.

```python
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```
- The color image is converted to grayscale. Grayscale images have only one channel (intensity) instead of three channels (red, green, and blue), making them easier to process.

```python
    # Normalize the image
    normalized_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
```
- The grayscale image values are normalized to a range of 0 to 255. This helps in enhancing the contrast and visibility of the image.

```python
    # Apply adaptive thresholding
    adaptive_thresh_img = cv2.adaptiveThreshold(normalized_img, 127, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
```
- Adaptive thresholding is applied to the normalized image. This technique adjusts the threshold dynamically for different regions of the image, which is useful for images with varying lighting conditions.

```python
    # Apply Unsharp Masking for auto-sharpness control
    blurred_img = cv2.GaussianBlur(adaptive_thresh_img, (5, 5), 0)

    a = .16  # contrast = 5. Contrast control ( 0 to 127)
    b = 60   # brightness = 2. Brightness control (0-100)
    c = 0
    sharp_img = cv2.addWeighted(adaptive_thresh_img, a, blurred_img, c, b)
```
- Unsharp Masking is used to enhance the sharpness of the image. It involves subtracting a blurred version of the image from the original to highlight edges and details.

```python
    # Apply Non-Local Means Denoising
    denoised_img = cv2.fastNlMeansDenoising(sharp_img, None, h=5, searchWindowSize=17)
```
- Non-Local Means Denoising is applied to reduce noise in the image while preserving details.

```python
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.55, tileGridSize=(8, 8))  # 16x16
    enhanced_img = clahe.apply(denoised_img)
```
- Contrast Limited Adaptive Histogram Equalization (CLAHE) is used to enhance the contrast of the image. It improves the visibility of details in both dark and light areas.

```python
    return enhanced_img
```
- The enhanced palm print image is returned as the output of the function.

```python
# Example usage:
enhanced_image = enhance_palm_print("hand_Scan_H.jpg")
```
- This line demonstrates how to use the `enhance_palm_print` function. It loads an image named "hand_Scan_H.jpg" and applies the enhancement process.

```python
# Display the enhanced image
plt.figure(figsize=(8, 8))
plt.title("Enhanced Image")
plt.imshow(enhanced_image, cmap='gray')
plt.axis('off')
plt.show()
```
- Finally, the enhanced image is displayed using matplotlib. The image is shown without axis labels, and a title "Enhanced Image" is added.

## Example Usage
To use this code, replace `"hand_Scan_H.jpg"` with the file path of your own palm print image. Running the code will enhance the image and display the result.
```

You can save the above content in a `.md` file and include it in your GitHub repository to provide a detailed explanation of the code.
