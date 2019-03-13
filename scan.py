from transform import four_point_transform
import cv2, imutils
from imgEnhance import Enhancer
import numpy as np
import math

def preProcess(image):

    ratio = image.shape[0] / 500.0
    image = imutils.resize(image, height = 500)

    grayImage  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gaussImage = cv2.GaussianBlur(grayImage, (5, 5), 0)    # 5,5,0

    edgedImage = cv2.Canny(gaussImage, 75, 200)  #75,200

    cnts = cv2.findContours(edgedImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)  # Calculating contour circumference
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    return  screenCnt, ratio

#限制读数在100-200之间
def LimitedMinMax(num, max, min):
    if num > max:
        return max
    elif num < min:
        return min
    else:
        return num

#指针角度及其数值
def getPointerAngleAndNum(img):

    num_dic = {"min": 100.0, "max": 200.0}
    angle_dic = {"min": 12.2, "max": 45.0}

    grayImage  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gaussImage = cv2.GaussianBlur(grayImage, (5, 5), 0)  #5,5,0

    edgedImage = cv2.Canny(gaussImage, 70, 300, apertureSize = 3)  #70,300

    # 霍夫变换
    minLineLength = 500
    maxLineGap = 20
    lines = cv2.HoughLinesP(edgedImage, 1, np.pi/180, 50, minLineLength, maxLineGap) #20

    rotate_angle = None
    num = None

    if lines is not None:
        for x1, y1, x2, y2 in lines[0]:
            if (x2 - x1) == 0 :
                continue
            t = float(y2 - y1) / (x2 - x1)

            # 指针的旋转角大小
            rotate_angle = math.degrees(math.atan(t))

            #指针旋转角对应的读数
            num = LimitedMinMax(num_dic["min"] + (rotate_angle - angle_dic["min"]) / (angle_dic["max"] - angle_dic["min"]) * (num_dic["max"] - num_dic["min"]), 200.0, 100.0)

            print("指针旋转角: %.2f 指针读数: %.2f" %(rotate_angle, num))

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow("line", imutils.resize(img, height = 250))

    return rotate_angle, num


#显示当前图像
def showImage(video = 1):
    # 记录上一次的指针读数值
    tmp = None

    # 读视频的每一帧
    vc = cv2.VideoCapture(video)  # 读入视频文件
    rval = vc.isOpened()

    while rval:  # 循环读取视频帧
        rval, frame = vc.read()
        if rval:

            screenCnt, ratio = preProcess(frame)

            if screenCnt is None:
                continue

            warped = four_point_transform(frame, screenCnt.reshape(4, 2) * ratio)

            enhancer = Enhancer()
            enhancedImg = enhancer.gamma(warped, 1.63)

            rotate_angle, num = getPointerAngleAndNum(warped)
            font = cv2.FONT_HERSHEY_DUPLEX

            if rotate_angle is not None and num is not None:
                img = cv2.putText(frame, str('%.2f' % (num)), (20, 40), font, 0.8, (0, 0, 255), 1)
                cv2.imshow("frame", imutils.resize(img, height = 400))
                tmp = num
            elif tmp is not None:  # 使用最近一次的指针读数作为未识别出时的指针读数
                img = cv2.putText(frame, str('%.2f' % (tmp)), (20, 40), font, 0.8, (0, 0, 255), 1)
                cv2.imshow("frame", imutils.resize(img, height = 400))
            else:
                cv2.imshow("frame", imutils.resize(frame, height = 400))

            key = cv2.waitKey(30)
            if key == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":

    video = "D:/test/pointer_45.avi"
    #video = "D:/test/pointer.avi"
    showImage(video)
    #showImage()



