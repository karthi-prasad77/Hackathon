# Object detection
# import nesecarry libraries
import cv2
import matplotlib.pyplot as plt
from matplotlib import ft2font

config_file = './yolo_v3/mobile_net_v3.pbtxt'
frozen_model = './yolo_v3/frozen_v3.pb'

# tensorflow object detection model
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# coco dataset
classLabels = []

filename = './yolo_v3/yolo.txt'

with open(filename, 'rt') as img:
    classLabels = img.read().rstrip('\n').split('\n')

print("Number of classes")
print(len(classLabels))
print("Class Labels")
print(classLabels)


# model training
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Reading image
image = cv2.imread('./yolo_v3/image.jpg')
plt.imshow(image)

# convet images from BGR to RGB
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# object detection
ClassIndex, Confidence, bbox = model.detect(image, confThreshold = 0.5)

# plotting boxes
font = 3
fonts = cv2.FONT_HERSHEY_PLAIN

for ClassInd, conf, boxes in zip(ClassIndex.flatten(), Confidence.flatten(), bbox):
    cv2.rectangle(image, boxes, (0, 255, 0), 3)   # for RGB channels
    cv2.putText(image, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), fonts, fontScale = font, color = (0, 0, 255), thickness = 4)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))