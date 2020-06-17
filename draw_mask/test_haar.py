import cv2 as cv

cap = cv.VideoCapture(0)

while True:
    ret, img = cap.read()
    faceCascade= cv.CascadeClassifier("xml/thumb.xml")
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    good = faceCascade.detectMultiScale(imgGray, 1.1, 10)
    for (x,y,w,h) in good:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv.imshow('thumb', img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()



 
# faceCascade= cv2.CascadeClassifier("xml/thumb.xml")
# img = cv2.imread('7.jpg')
# imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
# good = faceCascade.detectMultiScale(img,1.1,4)
# print(good)
 
# for (x,y,w,h) in good:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
 
# cv2.imshow("Result", img)
# cv2.waitKey(0)
