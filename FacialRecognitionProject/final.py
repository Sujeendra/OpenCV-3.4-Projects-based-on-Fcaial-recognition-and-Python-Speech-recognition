import speech_recognition as sr
import cv2
import os

import numpy as np

# this is called from the background thread
def callback(recognizer, audio):
    # received audio data, now we'll recognize it using Google Speech Recognition
    try:
        outp=recognizer.recognize_google(audio)
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        
        global val#global val will be read in main while loop
        if (outp!="can you hear me" and outp!="exit" and outp!="very good" and outp!="excellent" and outp!="can you see me" and outp!="who am I"):
                print("you said: " + outp)
        if (outp=="can you hear me"):
                print ("Yes")
        if (outp=="exit"):#this works
                val=1
                print ("Exiting...")
		
        if (outp=="put text"):
                val=2
        if (outp=="refresh"):
                val=0
        if (outp=="can you see me"):
                val=8
                print ("Yes")
        if (outp=="who am I"):
                print ("Admin")
                val=9
        if (outp=="very good" or outp=="excellent"):
                print ("Thank You")
		#stop_listening() # calling this function requests that the background listener stop listening
    except sr.UnknownValueError:
        print("could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

r = sr.Recognizer()
m = sr.Microphone(device_index=6)
cap = cv2.VideoCapture(1)
cap.set(3, 640) # set video widht
cap.set(4, 480) # set video height
imgsq=cv2.imread("Admin2.jpg")

# Define min window size to be recognized as a face
minW = 0.1*cap.get(3)
minH = 0.1*cap.get(4)
val=0
imgsq=cv2.imread("/home/sujeendra/FacialRecognitionProject/Admin2.jpg")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Sujeendra[Admin]', 'x', 'y', 'Z', 'W'] 

with m as source:
    r.adjust_for_ambient_noise(source) # we only need to calibrate once, before we start listening

# start listening in the background (note that we don't have to do this inside a `with` statement)
stop_listening = r.listen_in_background(m, callback)

# `stop_listening` is now a function that, when called, stops background listening

# do some other computation for 5 seconds, then stop listening and keep doing other computations
import time
#for x in range(50): time.sleep(0.1) # we're still listening even though the main thread is doing other things
#stop_listening() # calling this function requests that the background listener stop listening
while True: 
	#print "out"
        ret,img=cap.read()
	#print val
        if(val==2):
                cv2.putText(img,"I can hear you",(30,80),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,255),2)
        if(val==8):

                faces = face_cascade.detectMultiScale(img, 1.3, 5)
                for (x,y,w,h) in faces:
        		#cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                        roi_color = img[y:y+h, x:x+w]
                        rect=img[int(y+h/2-100):int(y+h/2+100),int(x+w/2-100):int(x+w/2+100)]
                        cv2.addWeighted(rect,1,imgsq,1,0,rect)
			
        if(val==9):
                img = cv2.flip(img, -1) # Flip vertically
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale( 
                   gray,
                   scaleFactor = 1.2,
                   minNeighbors = 5,
                   minSize = (int(minW), int(minH)),
                   )

                for (x,y,w,h) in faces:
                            

                        rect=gray[x:x+w,y:y+h]
                        cv2.addWeighted(rect,0.7,imgsq,0.3,0)

                        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
                        if (confidence < 100):
                            id = names[id]
                            confidence = "  {0}%".format(round(100 - confidence))
                        else:
                            id = "unknown"
                            confidence = "  {0}%".format(round(100 - confidence))
        
                        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
                        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
			
         
        if(cv2.waitKey(10) & 0xFF == ord('b')):
                break
        if(val==1):
                break
	
	#cv2.imshow('sq',imgsq)
        cv2.imshow('Video', img)
	#time.sleep(0.5)
