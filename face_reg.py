import numpy as np
import cv2
from PIL import Image
from numpy.dual import det
import pyautogui
sayac=1

dosya_yaz=open("deneme","a")

for sayac in range(1,2115):
    try:
        def detectFromImage(image):
            face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            img = cv2.imread(image)
            faces = face_cascade.detectMultiScale(img,1.5,6)
            for (x, y, w, h) in faces:
                img=cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3) #red color
                roi_color = img[y:y + h, x:x + w]
                cropped=img[y:y+h, x:x+w]
            cv2.imshow(str(sayac),cropped)
            pyautogui.press('enter')
           # cv2.imshow("face detection", img)
            print(sayac)
            dosya_yaz.write(str(sayac)+",")
            cv2.imwrite(str(sayac)+".jpg",cropped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        detectFromImage(str(sayac)+".jpg")
    except UnboundLocalError:
        print("Bozuk dosya:"+str(sayac))
        continue
dosya_yaz.close()


