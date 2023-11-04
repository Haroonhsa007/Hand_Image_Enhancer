from matplotlib import pyplot as plt
import cv2
import numpy as np

# ... (your code for the PalmPrintEnhancer class)
class PalmPrintEnhancer:
    def __init__(self, image):
        self.image = image

    def preprocess(self):
        # Apply noise reduction
        denoised_image = cv2.medianBlur(self.image, 5)

        # Convert denoised image to CV_8UC1
        denoised_image = denoised_image.astype(np.uint8)

        # Enhance contrast
        contrasted_image = cv2.equalizeHist(denoised_image)

        # Normalize image
        normalized_image = contrasted_image / 255.0

        return normalized_image

    def extract_features(self):
        # Extract palm lines
        palm_lines = extract_palm_lines(self.image)

        # Extract palm print minutiae
        minutiae = extract_palm_print_minutiae(self.image)

        return palm_lines, minutiae

    def enhance(self, palm_lines, minutiae):
        # Enhance ridges
        enhanced_ridges = enhance_ridges(self.image, palm_lines)

        # Enhance minutiae
        enhanced_minutiae = enhance_minutiae(minutiae)

        return enhanced_ridges, enhanced_minutiae

    def postprocess(self, enhanced_ridges, enhanced_minutiae):
        # Smooth image
        smoothed_image = cv2.GaussianBlur(enhanced_ridges, (5, 5), 0)

        # Binarize image
        binary_image = cv2.threshold(smoothed_image, 0.5, 1.0, cv2.THRESH_BINARY)[1]

        return binary_image

    def enhance_palm_print(self):
        # Preprocess image
        preprocessed_image = self.preprocess()

        # Extract features
        palm_lines, minutiae = self.extract_features()

        # Enhance palm print
        enhanced_ridges, enhanced_minutiae = self.enhance(palm_lines, minutiae)

        # Postprocess image
        enhanced_palm_print = self.postprocess(enhanced_ridges, enhanced_minutiae)

        return enhanced_palm_print

# Example usage:

# Load palm print image
image = cv2.imread("hand_Scan_H.jpg")

# Create palm print enhancer
enhancer = PalmPrintEnhancer(image)

# Enhance palm print
enhanced_palm_print = enhancer.enhance_palm_print()

# Display enhanced palm print
plt.imshow(enhanced_palm_print, cmap='gray')
plt.title('Enhanced Palm Print')
plt.axis('off')
plt.show()
