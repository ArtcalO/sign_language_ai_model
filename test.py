import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time



cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("models/keras_model.h5", "models/labels.txt")


offset = 20
imgSize = 300
counter = 0

folder = "data/C"

labels = ["A","B","C"]

while True:
	success, img = cap.read()
	imgOutput = img.copy()
	hands,img = detector.findHands(img, draw=False)
	
	if hands:
		hand = hands[0]

		x,y,w,h = hand['bbox']

		imgWhiteTemplate = np.ones((imgSize, imgSize,3), np.uint8)*255
		imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
		imgCropShape = imgCrop.shape


		aspectRatio = h/w

		if(aspectRatio>1):
			k = imgSize/h
			wCalc = math.ceil(k*w)
			imageResized = cv2.resize(imgCrop, (wCalc, imgSize))
			imageResizedShape = imageResized.shape
			
			wGap = math.ceil((imgSize-wCalc)/2)

			imgWhiteTemplate[:,wGap:wCalc+wGap] = imageResized
			prediction, index=classifier.getPrediction(imgWhiteTemplate)
			print(prediction, index)

		else:
			k = imgSize/w
			hCalc = math.ceil(k*h)
			imageResized = cv2.resize(imgCrop, (imgSize, hCalc))
			imageResizedShape = imageResized.shape
			
			hGap = math.ceil((imgSize-hCalc)/2)

			imgWhiteTemplate[hGap:hCalc+hGap,:] = imageResized
			prediction, index=classifier.getPrediction(imgWhiteTemplate)
			print(prediction, index)

		cv2.putText(imgOutput, labels[index], (x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(225,0,255),2)
		cv2.imshow("ImageCroped", imgCrop)
		cv2.imshow("ImgWhite", imgWhiteTemplate)

	cv2.imshow('Image', imgOutput)
	key = cv2.waitKey(1)

	