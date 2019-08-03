# coding=utf-8
import math
import time
from naoqi import ALProxy
import almath
from PIL import Image
import cv2
import numpy as np


class ComputeHitBallPosition(object):
    """
    调用computeHitBallPosition2方法，获取球洞坐标，红球坐标， 击球点坐标
    """

    def __init__(self, robotIp, port, landmarkTheoreticalSize=0.105, HitBallD=0.10):
        self.robotIp = robotIp
        self.port = port
        self.landmarkTheoreticalSize = landmarkTheoreticalSize
        self.HitBallD = HitBallD
        self.cameraAngleX = 60.97 * math.pi / 180
        self.cameraAngleY = 47.64 * math.pi / 180

    def __RedBallDetect(self, img, imageWidth, imageHeight):
        """
        :param img: cv2 Image
        :param imageWidth: image Width
        :param imageHeight: image Height
        :return: (1,center) or (None,None), Red Ball center
        """
        flag = 0
        hue_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gaus = cv2.GaussianBlur(hue_image, (7, 7), 1.5)
        # 用颜色分割图像
        low_range = np.array([0, 123, 100])
        high_range = np.array([5, 255, 255])
        th = cv2.inRange(gaus, low_range, high_range)

        # 形态学运算，膨胀
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)

        # Hough Circle
        circles = cv2.HoughCircles(dilated, cv2.HOUGH_GRADIENT, 1, 100, param1=15, param2=7, minRadius=10,
                                   maxRadius=100)

        # 绘制
        center = (0, 0)
        if circles is not None:
            x, y, radius = circles[0][0]
            center = (x, y)
            cv2.circle(img, center, radius, (0, 255, 0), 2)
            flag = 1

        cv2.line(img, (imageWidth / 2, 0), (imageWidth / 2, imageHeight), (255, 0, 0), 1)
        cv2.line(img, (0, imageHeight / 2), (imageWidth, imageHeight / 2), (255, 0, 0), 1)
        cv2.imshow('Result', img)
        cv2.imshow('Binaryzation', th)
        cv2.waitKey()

        if flag:
            return 1, center
        if not flag:
            return None, None

    def __computeRedBall(self, ballR=0.018):
        """
        :return:Red Ball x, y, distance
        """
        camProxy = ALProxy("ALVideoDevice", self.robotIp, 9559)
        motionProxy = ALProxy("ALMotion", self.robotIp, 9559)
        camProxy.setActiveCamera(1)
        cameraId = 1
        resolution = 2  # VGA 解析度
        colorSpace = 11  # RGB
        fps = 5
        subscriberID = "subscriberID"

        subscriberID = camProxy.subscribeCamera(subscriberID, cameraId, resolution, colorSpace, fps)
        naoImage = camProxy.getImageRemote(subscriberID)

        camProxy.releaseImage(subscriberID)
        camProxy.unsubscribe(subscriberID)
        imageWidth = naoImage[0]
        imageHeight = naoImage[1]
        image_bytes = naoImage[6]

        img_np = Image.frombytes("RGB", (imageWidth, imageHeight), image_bytes)
        cameraPosition = motionProxy.getPosition("CameraBottom", 2, True)
        cameraX = cameraPosition[0]
        cameraY = cameraPosition[1]
        print("cameraX: ", cameraX)
        cameraHeight = cameraPosition[2]
        print("cameraHeight: ", cameraHeight)

        # PIL image 转cv2 image
        open_cv_image = np.array(img_np)
        img1 = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        RedBall, center = self.__RedBallDetect(img1, imageWidth, imageHeight)

        if not RedBall:
            print("none")
            return None, None, None
        else:
            print("center: {}".format(center))
            ballAngleX = cameraX + (imageWidth / 2 - center[0]) / imageWidth * self.cameraAngleX
            print("ballAngleX: ", ballAngleX)
            ballAngleY = cameraY + (center[1] - imageHeight / 2) / imageHeight * self.cameraAngleY
            print("ballAngleY: ", ballAngleY)
            ballDistance = math.tan(ballAngleY) * (cameraHeight - ballR)
            print("ballDistance: ", ballDistance)
            ballX = ballDistance * math.sin(ballAngleX)
            ballY = ballDistance * math.cos(ballAngleY)
            return ballX, ballY, ballDistance

    def __computeMarkDistance(self):
        """
        :return: HoleX, HoleY, MarkDistance
        """
        currentCamera = "CameraTop"
        memoryProxy = ALProxy("ALMemory", self.robotIp, self.port)
        landmarkProxy = ALProxy("ALLandMarkDetection", self.robotIp, self.port)
        landmark = landmarkProxy.subscribe("landmarkTest")
        if landmark is not None:
            print("subscribe Fail")
        markData = memoryProxy.getData("LandmarkDetected")
        while markData is None or len(markData) == 0:
            time.sleep(0.01)
            markData = memoryProxy.getData("LandmarkDetected")
        landmarkProxy.unsubscribe("landmarkTest")
        wzCamera = markData[1][0][0][1]
        wyCamera = markData[1][0][0][2]
        angularSize = markData[1][0][0][3]
        distanceFromCameraToLandmark = self.landmarkTheoreticalSize / (2 * math.tan(angularSize / 2))
        motionProxy = ALProxy("ALMotion", self.robotIp, self.port)
        transform = motionProxy.getTransform(currentCamera, 2, True)
        transformList = almath.vectorFloat(transform)
        robotToCamera = almath.Transform(transformList)
        cameraToLandmarkRotationTransform = almath.Transform_from3DRotation(0, wyCamera, wzCamera)
        cameraToLandmarkTranslationTransform = almath.Transform(distanceFromCameraToLandmark, 0, 0)
        robotToLandmark = robotToCamera * cameraToLandmarkRotationTransform * cameraToLandmarkTranslationTransform
        x = robotToLandmark.r1_c4
        y = robotToLandmark.r2_c4
        print "x " + str(robotToLandmark.r1_c4) + " (in meters)"
        print "y " + str(robotToLandmark.r2_c4) + " (in meters)"
        print "z " + str(robotToLandmark.r3_c4) + " (in meters)"
        distance = math.sqrt(robotToLandmark.r1_c4 ** 2 + robotToLandmark.r2_c4 ** 2)
        return x, y, distance

    def computeHitBallPosition2(self):
        """
        调用此方法，获取球洞坐标，红球坐标， 击球点坐标
        :return: A dict, HoleX, HoleY, ballX, ballY, HitX, HitY
        """
        hb = self.HitBallD  # The distance from the hitting point to the ball
        HoleX, HoleY, MarkD = self.__computeMarkDistance()

        BallX, BallY, BallD = self.__computeRedBall()
        k = (HoleY - BallY) / (HoleX - BallX)

        HitX = math.sqrt(abs(math.sqrt(hb) / (k + 1))) + BallX  # Hit ball positions
        HitY = k * (math.sqrt(abs(math.sqrt(hb) / (k + 1)))) + BallY

        return {"HoleX": HoleX, "HoleY": HoleY, "BallX": BallX, "BallY": BallY, "HitX": HitX, "HitY": HitY}


if __name__ == '__main__':
    HitBall = ComputeHitBallPosition("192.168.1.104", 9559)
    HitBallDict = HitBall.computeHitBallPosition2()
    Holex = HitBallDict["HoleX"]
    Holey = HitBallDict["HoleY"]
    ballx = HitBallDict["BallX"]
    bally = HitBallDict["BallY"]
    Hitx = HitBallDict["HitX"]
    Hity = HitBallDict["HitY"]
    print("\nHoleX:{}\nHoleY:{}\nballX:{}\nballY:{}\nHitX:{}\nHitY:{}".format(Holex, Holey, ballx, bally, Hitx, Hity))
