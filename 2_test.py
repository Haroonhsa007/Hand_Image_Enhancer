import cv2

# Load the image
image = cv2.imread('hand_Scan_H.jpg')

# Initialize HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Detect hands
boxes, _ = hog.detectMultiScale(image)

# Draw boxes around hands
for (x, y, w, h) in boxes:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result
cv2.imshow('Hand Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
