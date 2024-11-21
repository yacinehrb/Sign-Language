import tensorflow as tf 
import numpy as np
import operator
import pyttsx3

import cv2

model= tf.keras.models.load_model("C:\PROJECTS\AI\SIGN LANGUAGE TRANSLATOR\My_model\CNN_SIGN_LANGUAGE")
categories = {0: 'A', 1: 'B', 2: 'C',3:'D',4:'E', 5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',
             11:'L',12:'M',13:'N',14:'NOTHING',15:'O',16:'P',17:'Q',18:'R',19:'S',20:'SPACE',
             21:'T',22:'U',23:'V',24:'W',25:'X',26:'Y'}

cap = cv2.VideoCapture(0)
text_1=''
word=''

def speak(text):
    engine= pyttsx3.init()	
    engine.setProperty("rate",120)
    voices=engine.getProperty("voices")
    engine.setProperty("voice", voices[1].id)
    engine.say(text)
    engine.runAndWait()


while True:
    blank=np.zeros((310,310), dtype='uint8')
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    roi_1 = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi_1, (64,64)) 
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    test_image = cv2.Canny(roi,15,70)
    cv2.imshow("test", test_image)
    result =model.predict(test_image.reshape(1,64,64,1))

    prediction = {'A': result[0][0], 
                  'B': result[0][1],
                  'C': result[0][2],
                  'D': result[0][3],
                  'E': result[0][4],
                  'F': result[0][5],
                  'G': result[0][6],
                  'H': result[0][7], 
                  'I': result[0][8],
                  'J': result[0][9],
                  'K': result[0][10],
                  'L': result[0][11],
                  'M': result[0][12],
                  'N': result[0][13],
                  'NOTHING': result[0][14],
                  'O': result[0][15],
                  'P': result[0][16],
                  'Q': result[0][17],
                  'R': result[0][18],
                  'S': result[0][19],
                  ' ': result[0][20],
                  'T': result[0][21],
                  'U': result[0][22],
                  'V': result[0][23],
                  'W': result[0][24],
                  'X': result[0][25],
                  'Y': result[0][26]
                  }
    
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    #cv2.putText(frame,'PREDICTION:',(10, 80),1, cv2.FONT_HERSHEY_DUPLEX, (0,0,255),1)
    cv2.putText(roi_1, prediction[0][0], (20, 60),1, cv2.FONT_HERSHEY_DUPLEX, (0,0,255),1)

    cv2.imshow('roi_1',roi_1)
    #cv2.imshow("Frame", frame)
    interrupt = cv2.waitKey(10)
    cv2.putText(blank,"TEXT:",(10,50),1,cv2.FONT_HERSHEY_DUPLEX,(255,0,0),1)
    
    if interrupt & 0xFF==ord('c'): # Press 'c' to write on the screen 
        text_0=prediction[0][0]
        if text_0 == "NOTHING":
            word = word[:-1]
            cv2.putText(blank,word,(10,90),1,cv2.FONT_HERSHEY_PLAIN,(255,255,255),2)
            cv2.imshow("text",blank)
            
        elif text_0!='':
            word=word+text_0
            cv2.putText(blank,word,(10,90),1,cv2.FONT_HERSHEY_PLAIN,(255,255,255),2)
            text_1=text_0
            cv2.imshow("text",blank)

    if interrupt & 0xFF==ord('p'): # Press 'p' to play the displayed text
        speak(word)
    
    if interrupt & 0xFF==ord('d'): # Press 'd' to delete all the text
        word=''
        cv2.putText(blank,word,(10,90),1,cv2.FONT_HERSHEY_PLAIN,(255,255,255),2)
        cv2.imshow("text",blank)

    screen= cv2.imread('C:\PROJECTS\AI\SIGN LANGUAGE TRANSLATOR\Picture.png')
    cv2.imshow('sign language',screen)

    if interrupt & 0xFF == 27: # esc key
        break
cap.release()
cv2.destroyAllWindows()