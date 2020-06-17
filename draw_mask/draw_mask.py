import cv2 as cv
import numpy as np

# 打开摄像头相关
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 600)
# cap.set(cv.CAP_PROP_FPS, 40)
# cap.set(3, 600)
# cap.set(4, 500)
# 收集画的点图，形成轨迹
points_line = []
# 记录当检测到出现暂停手势时，停止绘制图像(1:停止)
stop_thumb = 0
# 记录所画面具的中点
x_center, y_center = 0, 0
draw_num = 0

my_hsvcolor = [[129, 124, 101, 180, 255, 255], # purple
                [51, 76, 70, 68, 255, 255], # green
                [129, 65, 66, 160, 253, 255]] # red

hsvcolor_value = [[218, 112, 214], # 紫                
                    [0, 128, 0], # 绿
                    [255, 105, 180]] # 红


def getRightColor():
    cv.namedWindow('red')
    cv.resizeWindow('red', (500, 500))
    cv.createTrackbar('HUE MIN', 'red', 0, 180, nothing)
    cv.createTrackbar('SAT MIN', 'red', 0, 255, nothing)
    cv.createTrackbar('VALUE MIN', 'red', 0, 255, nothing)
    cv.createTrackbar('HUE MAX', 'red', 180, 180, nothing)
    cv.createTrackbar('SAT MAX', 'red', 255, 255, nothing)
    cv.createTrackbar('VALUE MAX', 'red', 255, 255, nothing)
    # 得到四条轨迹的当前位置
    hue_min = cv.getTrackbarPos('HUE MIN', 'red')
    sat_min = cv.getTrackbarPos('SAT MIN', 'red')
    value_min = cv.getTrackbarPos('VALUE MIN', 'red')
    hue_max = cv.getTrackbarPos('HUE MAX', 'red')
    sat_max = cv.getTrackbarPos('SAT MAX', 'red')
    value_max = cv.getTrackbarPos('VALUE MAX', 'red')
    print(hue_min, sat_min, value_min, hue_max, sat_max, value_max)
    mask = cv.inRange(imgHSV,np.array([hue_min, sat_min, value_min]),np.array([hue_max, sat_max, value_max]))
    # imgHSV[:] = [hue_min, sat_min, value_min]
    cv.imshow('color', mask)

def nothing(x):
    pass

def findcolor(img):
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # 这个是前期获取正确my_hsvcolor的函数
    # getRightColor()
    # 记录当前笔的颜色
    color_count = 0
    # 收集画的点图，形成轨迹
    # points_line = []

    for color in my_hsvcolor:
        cv.namedWindow('color{}'.format(color))
        cv.resizeWindow('color{}'.format(color), (500, 500))
        mask = cv.inRange(imgHSV,np.array(color[0:3]),np.array(color[3:6]))
        # cv.imshow('color{}'.format(color), mask)

        # 获取当前画笔头部点
        x, y = getCounters(mask, img)
        # print(x, y)
        # 将头部点的轨迹显示出来
        cv.circle(img, (x, y), 6, hsvcolor_value[color_count], cv.FILLED)
        # 画出轨迹
        if x!=0 and y!=0 and stop_thumb == 0:
            points_line.append([x, y, color_count])
            
        color_count += 1
        
    return points_line


def drawLines(points_line, img):
    for points in points_line:
        cv.circle(img, (points[0], points[1]), 6, hsvcolor_value[points[2]], cv.FILLED)
        

def getCounters(mask, img):
    counters, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # print(counters)
    # 画出轮廓
    # cv.drawContours(img, counters, -1, (0,255,0), 3)
    x, y, w, h = 0, 0, 0, 0
    for counter in counters:
        area = cv.contourArea(counter)
        if area >= 30:
            # 进行轮廓近似
            perimeter = cv.arcLength(counter, True)
            # epsilon = 0.1 * perimeter
            approx = cv.approxPolyDP(counter, 0.1 * perimeter, True)
            x,y,w,h = cv.boundingRect(approx)
    return x+w//2,y


# 检测到人脸时，使已经画好的画面跟随人脸移动
def faceShow(points_line, x_center, y_center, draw_num):
    for point in points_line:
        x_center += point[0]
        y_center += point[1]
        draw_num += 1
    return x_center//draw_num, y_center//draw_num


# 根据人脸位置改变已画好的图像的位置
def chDrawPosition(points_line, x_mid, y_mid, x2, y2, w2, h2):
    face_centerx = (x2 + (x2+w2))//2
    face_centery = (y2 + (y2+h2))//2
    deltax = x_mid - face_centerx
    deltay = y_mid - face_centery
    for point in points_line:
        point[0] = point[0]-deltax
        point[1] = point[1]-deltay
    return points_line


if not cap.isOpened():
    print('无法打开摄像头')
    exit()

while True:
    ret, img = cap.read()
    img = cv.flip(img, 1, dst=None)
    x2 = 0
    # imgResult = img.copy()
    # img = cv.GaussianBlur(img, (5,5), 0)
    # img = cv.medianBlur(img,5)

    # 检测暂停手势操作,面部识别
    thumb = cv.CascadeClassifier("xml/palm.xml")
    faceCascade= cv.CascadeClassifier("xml/haarcascade_frontalface_alt.xml")
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    good = thumb.detectMultiScale(imgGray, 1.1, 20)
    good_face = faceCascade.detectMultiScale(imgGray, 1.1, 20)
    # print(good)
    for (x,y,w,h) in good:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    for (x2,y2,w2,h2) in good_face:
        cv.rectangle(img,(x2,y2),(x2+w2,y2+h2),(255,191,0),2)

    if len(good)==0:
        stop_thumb = 0
    else:
        stop_thumb = 1

    # 找图像
    points_line = findcolor(img)
    # 出现人脸时改变points_line的坐标
    if len(points_line)>0 and x2>0:
        x_mid, y_mid = faceShow(points_line, x_center, y_center, draw_num)
        points_line = chDrawPosition(points_line, x_mid, y_mid, x2, y2, w2, h2)

        # print(x_mid, y_mid)
    # 画轨迹
    if len(points_line)>0:
        drawLines(points_line, img)

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv.imshow('my camera', img)
    # cv.imshow('my camera', imgResult)
    
    if cv.waitKey(1) == ord('q'):
        break

# 释放资源
cap.release()
cv.destroyAllWindows()

