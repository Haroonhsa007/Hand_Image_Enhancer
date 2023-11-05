``````python
# Palm Print Enhancement 
```ImEnM.py```
```

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



# Image Processing For Img enhancement and Feature Extraction

## Source 

` https://theailearner.com/category/image-processing/ `

## Canny Edge Detector

### Code 

Canny Edge Detector
In this blog, we will discuss one of the most popular algorithms for edge detection known as Canny Edge detection. It was developed by John F. Canny in 1986. It is a multi-stage algorithm that provides good and reliable detection. So, let’s discuss the main steps used in the Canny Edge detection algorithm using OpenCV-Python.

1. Noise Reduction
An edge detector is a high pass filter that enhances the high-frequency component and suppresses the low ones. Since both edges and noise are high-frequency components, the edge detectors tend to amplify the noise. To prevent this, we smooth the image with a low-pass filter. Canny uses a Gaussian filter for this.

Below is the code for this using OpenCV-Python

import cv2
import numpy as np
 
img = cv2.imread('D:/downloads/child.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# Gaussian Blurring
blur = cv2.GaussianBlur(gray,(5,5),0)

A larger filter reduces noise but worsens edge localization and vice-versa. Generally, 5×5 is a good choice but this may vary from image to image.

2. Finding Intensity Gradient of the Image
Next step is to find the edges using a Sobel operator. Sobel finds the gradients in both horizontal(Gx) and vertical(Gy) direction. Since edges are perpendicular to the gradient direction, using these gradients we can find the edge gradient and direction for each pixel as:


Below is the code for this using OpenCV-Python (Here, I’ve converted everything to 8-bit, it’s optional you can use any output datatype)

# Apply Sobelx in high output datatype 'float32'
# and then converting back to 8-bit to prevent overflow
sobelx_64 = cv2.Sobel(blur,cv2.CV_32F,1,0,ksize=3)
absx_64 = np.absolute(sobelx_64)
sobelx_8u1 = absx_64/absx_64.max()*255
sobelx_8u = np.uint8(sobelx_8u1)
 
# Similarly for Sobely
sobely_64 = cv2.Sobel(blur,cv2.CV_32F,0,1,ksize=3)
absy_64 = np.absolute(sobely_64)
sobely_8u1 = absy_64/absy_64.max()*255
sobely_8u = np.uint8(sobely_8u1)
 
# From gradients calculate the magnitude and changing
# it to 8-bit (Optional)
mag = np.hypot(sobelx_8u, sobely_8u)
mag = mag/mag.max()*255
mag = np.uint8(mag)
 
# Find the direction and change it to degree
theta = np.arctan2(sobely_64, sobelx_64)
angle = np.rad2deg(theta)

Clearly, we can see that the edges are still quite blurred or thick. Remember that an edge detector should output only one accurate response corresponding to the edge. Thus we need to thin the edges or in other words find the largest edge. This is done using Non-max Suppression.

3. Non-Max Suppression
This is an edge thinning technique. In this, for each pixel, we check if it is a local maximum in its neighborhood in the direction of gradient or not. If it is a local maximum it is retained as an edge pixel, otherwise suppressed.

For each pixel, the neighboring pixels are located in horizontal, vertical, and diagonal directions (0°, 45°, 90°, and 135°). Thus we need to round off the gradient direction at every pixel to one of these directions as shown below.


After rounding, we will compare every pixel value against the two neighboring pixels in the gradient direction. If that pixel is a local maximum, it is retained as an edge pixel otherwise suppressed. This way only the largest responses will be left.

Let’s see an example

Suppose for a pixel ‘A’, the gradient direction comes out to be 17 degrees. Since 17 is nearer to 0, we will round it to 0 degrees. Then we select neighboring pixels in the rounded gradient direction (See B and C in below figure). If the intensity value of A is greater than that of B and C, it is retained as an edge pixel otherwise suppressed.


Let’s see how to do this using OpenCV-Python

# Find the neighbouring pixels (b,c) in the rounded gradient direction
# and then apply non-max suppression
M, N = mag.shape
Non_max = np.zeros((M,N), dtype= np.uint8)
 
for i in range(1,M-1):
    for j in range(1,N-1):
       # Horizontal 0
        if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180) or (-22.5 <= angle[i,j] < 0) or (-180 <= angle[i,j] < -157.5):
            b = mag[i, j+1]
            c = mag[i, j-1]
        # Diagonal 45
        elif (22.5 <= angle[i,j] < 67.5) or (-157.5 <= angle[i,j] < -112.5):
            b = mag[i+1, j+1]
            c = mag[i-1, j-1]
        # Vertical 90
        elif (67.5 <= angle[i,j] < 112.5) or (-112.5 <= angle[i,j] < -67.5):
            b = mag[i+1, j]
            c = mag[i-1, j]
        # Diagonal 135
        elif (112.5 <= angle[i,j] < 157.5) or (-67.5 <= angle[i,j] < -22.5):
            b = mag[i+1, j-1]
            c = mag[i-1, j+1]           
            
        # Non-max Suppression
        if (mag[i,j] >= b) and (mag[i,j] >= c):
            Non_max[i,j] = mag[i,j]
        else:
            Non_max[i,j] = 0

Clearly, we can see that the edges are thinned but some edges are more bright than others. The brighter ones can be considered as strong edges but the lighter ones can actually be edges or they can be because of noise.

4. Hysteresis Thresholding
Non-max suppression outputs a more accurate representation of real edges in an image. But you can see that some edges are more bright than others. The brighter ones can be considered as strong edges but the lighter ones can actually be edges or they can be because of noise. To solve the problem of “which edges are really edges and which are not” Canny uses the Hysteresis thresholding. In this, we set two thresholds ‘High’ and ‘Low’.

Any edges with intensity greater than ‘High’ are the sure edges.
Any edges with intensity less than ‘Low’ are sure to be non-edges.
The edges between ‘High’ and ‘Low’ thresholds are classified as edges only if they are connected to a sure edge otherwise discarded.
Let’s take an example to understand


Here, A and B are sure-edges as they are above ‘High’ threshold. Similarly, D is a sure non-edge. Both ‘E’ and ‘C’ are weak edges but since ‘C’ is connected to ‘B’ which is a sure edge, ‘C’ is also considered as a strong edge. Using the same logic ‘E’ is discarded. This way we will get only the strong edges in the image.

This is based on the assumption that the edges are long lines.

Below is the code using OpenCV-Python.

First set the thresholds and classify edges into strong, weak or non-edges.

# Set high and low threshold
highThreshold = 21
lowThreshold = 15
 
M, N = Non_max.shape
out = np.zeros((M,N), dtype= np.uint8)
 
# If edge intensity is greater than 'High' it is a sure-edge
# below 'low' threshold, it is a sure non-edge
strong_i, strong_j = np.where(Non_max >= highThreshold)
zeros_i, zeros_j = np.where(Non_max < lowThreshold)
 
# weak edges
weak_i, weak_j = np.where((Non_max <= highThreshold) & (Non_max >= lowThreshold))
 
# Set same intensity value for all edge pixels
out[strong_i, strong_j] = 255
out[zeros_i, zeros_j ] = 0
out[weak_i, weak_j] = 75
For weak edges, if it is connected to a sure edge it will be considered as an edge otherwise suppressed.

M, N = out.shape
for i in range(1, M-1):
    for j in range(1, N-1):
        if (out[i,j] == 75):
            if 255 in [out[i+1, j-1],out[i+1, j],out[i+1, j+1],out[i, j-1],out[i, j+1],out[i-1, j-1],out[i-1, j],out[i-1, j+1]]:
                out[i, j] = 255
            else:
                out[i, j] = 0

OpenCV-Python
OpenCV provides a builtin function for performing Canny Edge detection

cv2.Canny(image, threshold1, threshold2[, apertureSize[, L2gradient]]])
# threshold1 and threshold2 are the High and Low threshold values
# apertureSize - Kernel size for the Sobel operator (Default is 3x3)
# L2gradient - whether to use L2norm for gradient magnitude calculation or not. Default is False that uses L1 norm.
Let’s take an example

import cv2
img = cv2.imread('D:/downloads/child.jpg',0)
edges = cv2.Canny(img,100,200,L2gradient=True)
Hope you enjoy reading.

If you have any doubt/suggestion please feel free to ask and I will do my best to help or improve myself. Good-bye until next time.

``````