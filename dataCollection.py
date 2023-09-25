import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
counter = 0

folder = "data/Y"

while True:
	success, img = cap.read()
	hands, img = detector.findHands(img)
	
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
		
		else:
			k = imgSize/w
			hCalc = math.ceil(k*h)
			imageResized = cv2.resize(imgCrop, (imgSize, hCalc))
			imageResizedShape = imageResized.shape
			
			hGap = math.ceil((imgSize-hCalc)/2)

			imgWhiteTemplate[hGap:hCalc+hGap,:] = imageResized
		

		cv2.imshow("ImageCroped", imgCrop)
		cv2.imshow("ImgWhite", imgWhiteTemplate)

	cv2.imshow('Image', img)
	key = cv2.waitKey(1)

	if key == ord('s'):
		counter+=1
		cv2.imwrite(f"{folder}/Img_{time.time()}.jpg", imgWhiteTemplate)
		print(counter)