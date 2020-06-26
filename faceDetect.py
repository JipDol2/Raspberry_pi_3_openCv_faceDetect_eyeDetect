import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontface.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
def detect(gray,img):
	#ret,img = cap.read()
	#img = cv2.flip(img,-1)
	#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(20,20))
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

		face_gray = gray[y:y+h,x:x+w]
		face_color = img[y:y+h,x:x+w]

		eyes = eyeCascade.detectMultiScale(face_gray,1.1,3)

		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(face_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	return img
	#cv2.imshow('vidio',img)
	#k = cv2.waitKey(30) & 0xFF
	#if k == 27:
	#	break
while True:
	ret,img = cap.read()
	img = cv2.flip(img,-1)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	canvas = detect(gray,img)
	cv2.imshow('video',canvas)

	k=cv2.waitKey(30) & 0xFF
	if k==27:
		break
cap.release()
cv2.destroyAllWindows()
