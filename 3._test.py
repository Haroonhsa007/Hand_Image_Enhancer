import cv2
import numpy as np

# Load the pre-trained hand detection model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'hand_model.caffemodel')

# Load the image
image = cv2.imread('hand_image.jpg')
(h, w) = image.shape[:2]

# Preprocess the image for the neural network
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

# Set the prepared image as input to the network
net.setInput(blob)
detections = net.forward()

# Loop over the detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    
    if confidence > 0.2:  # Confidence threshold
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Draw the bounding box
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# Display the result
cv2.imshow('Hand Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
