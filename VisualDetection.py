# coding=utf-8
import numpy as np
import cv2
import os
import time
import math
import almath
import codecs
from TargetFeature import HogFeature, ColorFeature
from Classifier import KNN
from naoqi import ALProxy
from functools import cmp_to_key
from datetime import datetime
import vision_definitions as vd
from ConfigureNao import *


class VisualBasis(ConfigureNao):
    """
    a basic class for visual task.
    """

    def __init__(self, robotIp, port=9559, cameraId=vd.kBottomCamera, resolution=vd.kVGA):
        """
        initilization.

        Args:
            IP: NAO's IP
            cameraId: bottom camera (1,default) or top camera (0).
            resolution: kVGA, default: 640*480)
        Return:
            none
        """
        super(VisualBasis, self).__init__(robotIp, port)
        self.cameraId = cameraId
        self.cameraName = "CameraBottom" if self.cameraId == vd.kBottomCamera else "CameraTop"
        self.resolution = resolution
        self.colorSpace = vd.kBGRColorSpace
        self.fps = 20
        self.frameHeight = 0
        self.frameWidth = 0
        self.frameChannels = 0
        self.frameArray = None
        self.cameraPitchRange = 47.64 / 180 * np.pi
        self.cameraYawRange = 60.97 / 180 * np.pi
        self.cameraProxy.setActiveCamera(self.cameraId)

    def updateFrame(self, client="python_client"):
        """
        get a new image from the specified camera and save it in self._frame.

        Args:
            client: client name.
        Return:
            none.
        """
        if self.cameraProxy.getActiveCamera() != self.cameraId:
            self.cameraProxy.setActiveCamera(self.cameraId)
            time.sleep(1)

        videoClient = self.cameraProxy.subscribe(client, self.resolution, self.colorSpace, self.fps)
        # print("videoClient: {}".format(videoClient))
        frame = self.cameraProxy.getImageRemote(videoClient)
        self.cameraProxy.unsubscribe(videoClient)
        try:
            self.frameWidth = frame[0]
            self.frameHeight = frame[1]
            self.frameChannels = frame[2]
            self.frameArray = np.frombuffer(frame[6], dtype=np.uint8).reshape([frame[1], frame[0], frame[2]])
        except IndexError:
            print("get image failed!")


class RedBallDetection(VisualBasis):
    """调用updateBallData，更新红球信息"""

    def __init__(self, robotIp, port=9559, cameraId=vd.kBottomCamera, resolution=vd.kVGA):
        super(RedBallDetection, self).__init__(robotIp, port, cameraId, resolution)
        self.ballData = {"centerX": 0, "centerY": 0, "radius": 0}
        self.ballPosition = {"disX": 0, "disY": 0, "angle": 0}
        self.ballRadius = 0.021
        self.ballImg = None

    # noinspection PyMethodMayBeStatic
    def __compute_score(self, rects):
        """计算score"""

        minHSV1 = np.array([0, 43, 46])
        maxHSV1 = np.array([10, 255, 255])
        minHSV2 = np.array([156, 43, 46])
        maxHSV2 = np.array([180, 255, 255])

        img = self.frameArray.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        Rects = list()
        # stime = datetime.now()
        for rect in rects:
            if isinstance(rect, list) is False:
                rect = rect.tolist()
            x, y, r = int(rect[0]), int(rect[1]), int(rect[2])
            H = hsv[y - r:y + r, x - r:x + r, 0]
            S = hsv[y - r:y + r, x - r:x + r, 1]
            V = hsv[y - r:y + r, x - r:x + r, 2]
            H = (np.where(H >= minHSV1[0]) and np.where(H <= maxHSV1[0])) or (
                    np.where(H >= minHSV2[0]) and np.where(H <= maxHSV2[0]))
            S = (np.where(S >= minHSV1[1]) and np.where(S <= maxHSV1[1])) or (
                    np.where(S >= minHSV2[2]) and np.where(S <= maxHSV2[2]))
            V = (np.where(V >= minHSV1[2]) and np.where(V <= maxHSV1[2])) or (
                    np.where(V >= minHSV2[2]) and np.where(V <= maxHSV2[2]))
            score = int((10000 * np.min([len(H), len(S), len(V)]) + 0.0000001) / (r ** 2 + 0.0000001))
            rect.append(score)
            Rects.append(rect)
        # print("hsv split time: {}s".format(datetime.now() - stime))
        return Rects

    def __nms(self, rects):
        """非极大值抑制"""
        rects = self.__compute_score(rects=rects)
        # stime = datetime.now()
        rects = sorted(rects, cmp=lambda a, b: a[3] - b[3])
        # print("sorted time: {}s".format(datetime.now() - stime))
        return rects[0]

    def __HoughDetection(self):
        """
        对图像进行霍夫圆检测
        首先图像预处理-----转hsv图像，阈值分割
        霍夫圆检测----调用opencv HoughCircles检测轮廓，返回可能的圆形轮廓列表
        :return:  rects of possible red ball
        """
        img = self.frameArray.copy()
        binImg = self.__preProcess(img)
        circles = cv2.HoughCircles(binImg, cv2.HOUGH_GRADIENT, 1, 100,
                                   param1=150, param2=25, minRadius=8, maxRadius=48)
        if circles is None:
            circles = []
            # print("no circle")
        else:
            circles = circles[0, :]
        return circles

    def __preProcess(self, img):
        """
        图像处理
        二值化：所有灰度大于或等于阀值的像素被判定为属于特定物体
        """
        HSVImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        minHSV1 = np.array([0, 43, 46])
        maxHSV1 = np.array([10, 255, 255])
        minHSV2 = np.array([156, 43, 46])
        maxHSV2 = np.array([180, 255, 255])

        # 二值化处理
        binImg1 = cv2.inRange(HSVImg, minHSV1, maxHSV1)
        binImg2 = cv2.inRange(HSVImg, minHSV2, maxHSV2)
        binImg = np.maximum(binImg1, binImg2)

        # 图像滤波处理（腐蚀，膨胀，高斯）
        binImg = self.__filter(binImg)
        return binImg

    # noinspection PyMethodMayBeStatic
    def __filter(self, img):
        """
        图像处理----在保证图像细节的基础上对图像进行降噪处理（噪声抑制）
        使用腐蚀膨胀消除噪点，高斯处理使图像平滑----线性平滑滤波消除高斯噪声
        :param img: 经过阈值处理的图像
        :return: 滤波后的图像
        """
        kernelErosion = np.ones((3, 3), np.uint8)
        kernelDilation = np.ones((3, 3), np.uint8)
        resImg = cv2.erode(img, kernelErosion, iterations=2)
        resImg = cv2.dilate(resImg, kernelDilation, iterations=3)
        resImg = cv2.GaussianBlur(resImg, (9, 9), 1.5)

        return resImg

    # noinspection PyMethodMayBeStatic
    def __reshapeBallRect(self, rawRect, numbers):
        """
        对当前矩形框进行四等分，为下一步计算颜色特征做准备
        :param rawRect:
        :param numbers:
        :return:
        """
        newRect = np.zeros((numbers, 4))
        initX, initY, endX, endY = rawRect[0], rawRect[1], rawRect[2], rawRect[3]  # 初始化参数

        newRect[0] = [initX, initY, initX + (endX - initX) / 2, initY + (endY - initY) / 2]
        newRect[1] = [initX + (endX - initX) / 2, initY, endX, initY + (endY - initY) / 2]
        newRect[2] = [initX, initY + (endY - initY) / 2, initX + (endX - initX) / 2, endY]
        newRect[3] = [initX + (endX - initX) / 2, initY + (endY - initY) / 2, endX, endY]

        return newRect

    # noinspection PyMethodMayBeStatic
    def __circle2Rect(self, circle, k=1):
        """转化球在图像中的矩形坐标"""
        centerX, centerY, radius = circle[0], circle[1], circle[2]
        initX, initY = int(centerX - int(k * radius)), int(centerY - int(k * radius))
        endX, endY = int(centerX + int(k * radius)), int(centerY + int(k * radius))

        return [initX, initY, endX, endY]

    # noinspection PyMethodMayBeStatic
    def __calColorFeature(self, img, number=16):
        """提取颜色特征"""
        color = ColorFeature(img, number)
        result = color.colorExtract(img)

        return np.round(result, 4)

    # noinspection PyMethodMayBeStatic
    def __calHOGFeature(self, img, cell_size):
        """提取HOG特征"""
        rectBallArea = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog = HogFeature(rectBallArea, cell_size)
        vector, img = hog.hog_extract()
        return np.round(vector[0], 4)

    def __result(self, isKnn=True):
        """
        传入图片，使用分类器
        然后对每一张图片进行目标检测
        并获得该目标区域的特征向量
        在放到分类器里面得到预测的结果，最后如果概率大于0.5，则认为是，否则为不是
        得到最终结果, 返回圆心，半径
        """
        Rects = []
        knn = KNN("data_ball.txt")
        srcImg = self.frameArray.copy()
        rects = self.__HoughDetection()
        if len(rects) is 0:
            # print("no rects")
            return []
        # 使用KNN分类
        if isKnn is False:
            Rect = self.__nms(rects)
            return Rect

        for rect in rects:  # 检测每个轮廓
            resultTotal = []
            x, y, r = rect[:3]

            # cv2.rectangle(srcImg, (x - r, y - r), (x + r, y + r), (0, 0, 255), 2)
            # cv2.imshow("img", srcImg)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            rect_ = [x, y, r]
            rect = self.__circle2Rect(rect)
            if rect[0] < 0 or rect[1] < 0 or rect[2] > 640 or rect[3] > 480:
                print("out of bound")
                continue
            newRects = self.__reshapeBallRect(rect, 4)  # 将球的矩形框分为四等分，分别进行特征提取

            for newRect in newRects:
                newInitX, newInitY = int(newRect[0]), int(newRect[1])
                newEndX, newEndY = int(newRect[2]), int(newRect[3])
                rectBallArea = srcImg[newInitY:newEndY, newInitX:newEndX, :]

                try:
                    resultColor = self.__calColorFeature(rectBallArea, 16)
                except Exception, err:
                    print(err)
                else:
                    cellSize = min(newEndX - newInitX, newEndY - newInitY)
                    resultHOG = self.__calHOGFeature(rectBallArea, cellSize / 2)
                    resultTotal.extend(resultColor)
                    resultTotal.extend(resultHOG)
            try:
                resultTotal = np.array(resultTotal).reshape(1, -1).astype('float64')
                classify = knn.classifyVector(resultTotal)
                if classify == 1:
                    Rects.append(rect_)
            except Exception, err:
                print(err)
        if len(Rects) is 0:
            return []
        Rect = self.__nms(Rects)

        return Rect

    def updateBallData(self, standState="standInit", client="python_client", isKnn=True):
        """
        更新红球信息
        """
        stime = time.time()
        self.updateFrame(client)
        with codecs.open("timeInfo.txt", 'a', encoding='utf-8') as fTime:
            fTime.write("take photo times: {:.2}s\n".format(time.time() - stime))
        cameraPosition = self.motionProxy.getPosition("CameraBottom", 2, True)
        cameraX = cameraPosition[0]
        cameraY = cameraPosition[1]
        cameraHeight = cameraPosition[2]
        img = self.frameArray.copy()
        imageHeight, imageWidth = self.frameHeight, self.frameWidth
        ball_rect = self.__result(isKnn)

        cameraYawRange = 60.97 * np.pi / 180
        cameraPitchRange = 47.64 * np.pi / 180

        if len(ball_rect) is not 0:
            centerX = int(ball_rect[0])
            centerY = int(ball_rect[1])
            radius = int(ball_rect[2])
            cv2.rectangle(img, (centerX - radius, centerY - radius), (centerX + radius, centerY + radius), (0, 0, 255),
                          2)
            self.ballImg = img

            bottomCameraDirection = {"standInit": 49.2, "standUp": 39.7}
            try:
                cameraDirection = bottomCameraDirection[standState]
                headPitches = self.motionProxy.getAngles("HeadPitch", True)
                headPitch = headPitches[0]
                headYaws = self.motionProxy.getAngles("HeadYaw", True)
                headYaw = headYaws[0]
                ballPitch = (centerY - imageHeight / 2) * cameraPitchRange / 480.0  # y (pitch angle)
                ballYaw = (imageWidth / 2 - centerX) * cameraYawRange / 640.0  # x (yaw angle)
                dPitch = (cameraHeight - self.ballRadius) / np.tan(
                    cameraDirection / 180 * np.pi + headPitch + ballPitch)
                dYaw = dPitch / np.cos(ballYaw)
                ballX = dYaw * np.cos(ballYaw + headYaw) + cameraX + 0.035
                ballY = dYaw * np.sin(ballYaw + headYaw) + cameraY
                ballYaw = np.arctan2(ballY, ballX)
                if standState == "standInit":
                    ky = 42.513 * ballX ** 4 - 109.66 * ballX ** 3 + 104.2 * ballX ** 2 - 44.218 * ballX + 8.5526
                    # ky = 12.604*ballX**4 - 37.962*ballX**3 + 43.163*ballX**2 - 22.688*ballX + 6.0526
                    ballY = ky * ballY
                    ballYaw = np.arctan2(ballY, ballX)

                self.ballData = {"centerX": centerX, "centerY": centerY, "radius": radius}
                self.ballPosition = {"disX": ballX, "disY": ballY, "angle": ballYaw}
            except KeyError:
                print("Error! unknown standState, please check the value of stand state!")

    def getBallPosition(self):
        """
        get ball position.

        Return:
            distance in x axis, distance in y axis and direction related to Nao.
        """
        disX = self.ballPosition["disX"]
        disY = self.ballPosition["disY"]
        angle = self.ballPosition["angle"]
        print("ball :", [disX, disY, angle])
        return [disX, disY, angle]

    def getBallInfoInImage(self):
        """
        get ball information in image.

        Return:
            a list of centerX, centerY and radius of the red ball.
        """
        centerX = self.ballData["centerX"]
        centerY = self.ballData["centerY"]
        radius = self.ballData["radius"]
        return [centerX, centerY, radius]

    def showBallPosition(self):
        """
        show and save ball data in the current frame.
        """
        if self.ballData["radius"] == 0:
            # print("no ball found.")
            print("ball postion = ", (self.ballPosition["disX"], self.ballPosition["disY"]))
            # cv2.imshow("ball position", self.frameArray)
        else:
            print("ballX = ", self.ballData["centerX"])
            print("ballY = ", self.ballData["centerY"])
            print("ball postion = ", (self.ballPosition["disX"], self.ballPosition["disY"]))
            print("ball direction = ", self.ballPosition["angle"] * 180 / 3.14)
            frameArray = self.frameArray.copy()
            cv2.circle(frameArray, (self.ballData["centerX"], self.ballData["centerY"]),
                       self.ballData["radius"], (250, 150, 150), 2)
            cv2.circle(frameArray, (self.ballData["centerX"], self.ballData["centerY"]),
                       2, (50, 250, 50), 3)
            # cv2.imshow("ball position", frameArray)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()


class StickDetection(VisualBasis):
    """调用compute_stickPosition，计算杆的方位角，距离"""

    def __init__(self, robotIp, port=9559, cameraId=vd.kTopCamera, resolution=vd.kVGA):
        super(StickDetection, self).__init__(robotIp, port, cameraId, resolution)
        self.boundRect = []
        self.stickAngle = 0.0  # rad
        self.stickH = 0.47
        self.stickDistance = 0
        self.minHSV = np.array([27, 55, 115])
        self.maxHSV = np.array([45, 255, 255])
        self.stickImg = None

    # noinspection PyMethodMayBeStatic
    def __reshapeStickRect(self, rawRect, numbers):
        """
        :param rawRect:  原始的矩形框
        :param numbers: 需要将原始矩形框进行等分的数量
        :return:  返回该矩形框等分后的列表
        """
        newRect = np.zeros((numbers, 4))
        initX, initY, endX, endY = rawRect[0], rawRect[1], rawRect[2], rawRect[3]  # 初始化参数

        # 找出每个小矩阵的顶点坐标
        for i in range(numbers):
            newRect[i][0] = initX
            newRect[i][1] = initY + ((endY - initY) / numbers) * i
            newRect[i][2] = endX
            newRect[i][3] = initY + ((endY - initY) / numbers) * (i + 1)

        return newRect  # 四等分矩阵信息

    # noinspection PyMethodMayBeStatic
    def __calColorFeature(self, img, number=16):
        """
        :param img: 原始图像
        :param number: 需要分成多少个区间
        :return:将特征向量除以总像素的个数得到归一化的结果
        """
        color = ColorFeature(img, number)
        result = color.colorExtract(img)

        return np.round(result, 4)

    # noinspection PyMethodMayBeStatic
    def __calHOGFeature(self, img, cellSize):
        """
        将图像分成小的连通区域， 选用梯度方向作为特征
        :param img:  原始图像
        :param cellSize: 细胞单元的大小
        :return:   采集细胞单元中各像素点的梯度边缘直方图， 将直方图组合起来构成特征描述器
        """
        rectStickArea = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog = HogFeature(rectStickArea, cellSize)
        vector, _ = hog.hog_extract()

        return np.round(vector[0], 4)

    # noinspection PyMethodMayBeStatic
    def __bilateral__filter(self, img):
        """
        双边滤波---保持边缘，降噪平滑
        """
        bilateral = cv2.bilateralFilter(img, 25, 12.5, 50)
        return bilateral

    # noinspection PyMethodMayBeStatic
    def __filter(self, binImg):
        """
        腐蚀，膨胀，高斯滤波
        :param binImg:二值图
        :return:处理结束的图
        """
        kernelErosion = np.ones((5, 5), np.uint8)
        kernelDilation = np.ones((5, 5), np.uint8)
        frameBin = cv2.erode(binImg, kernelErosion, iterations=1)
        frameBin = cv2.dilate(frameBin, kernelDilation, iterations=1)
        frameBin = cv2.GaussianBlur(frameBin, (9, 9), 0)

        return frameBin

    def __contoursDetection(self, img):  # 检测图像边界
        """
        :param img:  输入处理好的图像
        :return:     调用opencv, 检测矩形框
        """
        minPerimeter = 100
        minArea = 850

        # 预处理
        bilateral = self.__bilateral__filter(img)
        HSVImg = cv2.cvtColor(bilateral, cv2.COLOR_BGR2HSV)  # 转到HSV空间
        # 二值化处理
        minHSV = self.minHSV
        maxHSV = self.maxHSV
        binImg = cv2.inRange(HSVImg, minHSV, maxHSV)

        # 图像滤波处理（腐蚀，膨胀，高斯）
        frameBin = self.__filter(binImg)

        rects = []
        if cv2.__version__.split(".")[0] == "3":  # for OpenCV >= 3.0.0
            _, contours, _ = cv2.findContours(frameBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            contours, _ = cv2.findContours(frameBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return rects
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            if perimeter > minPerimeter and area > minArea:
                x, y, w, h = cv2.boundingRect(contour)
                rects.append([x, y, w, h])
        return rects

    # noinspection PyMethodMayBeStatic
    def __compute_score(self, rects):
        """计算score"""
        Rects = list()
        for rect in rects:
            if isinstance(rect, list) is False:
                rect = rect.tolist()
            w, h = rect[2], rect[3]
            rect.append(int(10000 * abs(float(w + 0.0000001) / (h + 0.0000001) - 0.1)))
            Rects.append(rect)
        return Rects

    def __nms(self, rects):
        """非极大值抑制"""
        rects = self.__compute_score(rects=rects)
        rects = sorted(rects, cmp=lambda a, b: a[4] - b[4])
        return rects[0]

    def __result(self, isKnn=True):
        img = self.frameArray
        Rects = []
        knn = KNN("data_stick.txt")
        # 返回检测到的矩形矩阵
        # 只需要检测边缘，返回边缘所在的矩阵，返回的轮廓
        rects = self.__contoursDetection(self.frameArray)

        if len(rects) == 0:
            # print("no rects")
            return []

        if isKnn is False:
            Rect = self.__nms(rects)
            return Rect

        for rect in rects:
            rect_ = rect[:4]
            resultTotal = []
            if rect[0] < 0 or rect[1] < 0 or rect[2] + rect[0] > 640 or rect[3] + rect[1] > 480:
                print("out of bound")
                continue
            rect[2] += rect[0]
            rect[3] += rect[1]
            newRects = self.__reshapeStickRect(rect, 4)

            for newRect in newRects:
                newInitX, newInitY = int(newRect[0]), int(newRect[1])
                newEndX, newEndY = int(newRect[2]), int(newRect[3])
                rectStickArea = img[newInitY:newEndY, newInitX:newEndX, :]

                resultColor = self.__calColorFeature(rectStickArea, 16)
                cellSize = min(newEndX - newInitX, newEndY - newInitY)
                resultHOG = self.__calHOGFeature(rectStickArea, cellSize / 2)
                resultTotal.extend(resultColor)
                resultTotal.extend(resultHOG)

            resultTotal = np.array(resultTotal).reshape(1, -1).astype('float64')
            classify = knn.classifyVector(resultTotal)

            if classify == 1:
                Rects.append(rect_)

        stick_rectangle = self.__nms(Rects)  # 非极大值抑制

        return stick_rectangle

    # noinspection PyMethodMayBeStatic
    def __distance_fixing(self, stickDistance):
        x = stickDistance
        a = -0.3165899349995387
        b = 1.150241967154712
        fx = (x - a) / b
        return fx

    # noinspection PyMethodMayBeStatic
    def updateStickData(self, client="stick", isKnn=True):
        """
        更新黄杆信息

        """
        stime = time.time()
        self.updateFrame(client)
        with codecs.open("timeInfo.txt", 'a', encoding='utf-8') as fTime:
            fTime.write("take photo times: {:.2}s\n".format(time.time() - stime))
        cameraPosition = self.motionProxy.getPosition("CameraBottom", 2, True)
        cameraHeight = cameraPosition[2]
        cameraAngles = self.motionProxy.getAngles("Head", True)
        cameraYaw, cameraPitch = cameraAngles
        img = self.frameArray
        imageHeight, imageWidth, _ = img.shape
        stick_rect = self.__result(isKnn)
        if len(stick_rect) is not 0:
            sx = stick_rect[0]
            sy = stick_rect[1]
            sw = stick_rect[2]
            sh = stick_rect[3]
            centerY = sy + sh / 2
            centerX = sx + 0.1 * sh / 2
            # 画出目标
            cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
            self.stickImg = img

            cameraRangeX = 60.97 * np.pi / 180
            cameraRangeY = 47.64 * np.pi / 180

            stickAngleX = cameraYaw + (imageWidth / 2.0 - centerX) / imageWidth * cameraRangeX
            stickAngleY = cameraPitch + (centerY - imageHeight / 2.0) / imageHeight * cameraRangeY

            stickDistance = (cameraHeight - self.stickH / 2.0) / np.tan(stickAngleY)
            stickDistance = self.__distance_fixing(stickDistance)

            self.stickDistance = stickDistance
            self.boundRect = [sx, sy, sw, sh]
            self.stickAngle = stickAngleX  # rad
        else:
            print("no stick")
            self.stickDistance = 0
            self.boundRect = []
            self.stickAngle = 0.0  # rad

    def showStickPosition(self):
        """
        show the stick  position in the current frame.
        """
        if len(self.boundRect) is 0:
            # print("no stick detected.")
            cv2.imshow("stick position", self.frameArray)
        else:
            [x, y, w, h] = self.boundRect
            frame = self.frameArray.copy()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # cv2.imshow("stick position", frame)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()


class LandMarkDetection(VisualBasis):
    """调用updateLandMarkData, 更新LandMark坐标，距离，角度"""

    def __init__(self, robotIp, port=9559, cameraId=vd.kTopCamera, landMarkSize=0.105):
        super(LandMarkDetection, self).__init__(robotIp, port)
        self.cameraId = cameraId
        self.cameraName = "CameraTop" if cameraId == vd.kTopCamera else "CameraBottom"
        self.landMarkSize = landMarkSize
        self.disX = 0
        self.disY = 0
        self.dist = 0
        self.yawAngle = 0
        self.cameraProxy.setActiveCamera(self.cameraId)

    def updateLandMarkData(self):
        """
        更新LandMark信息
        """
        currentCamera = "CameraTop"
        self.landmarkProxy.subscribe("landmark")
        markData = self.memoryProxy.getData("LandmarkDetected")
        if len(markData) is 0:
            self.disX = 0
            self.disY = 0
            self.dist = 0
            self.yawAngle = 0
            return
        wzCamera = markData[1][0][0][1]
        wyCamera = markData[1][0][0][2]
        angularSize = markData[1][0][0][3]
        distanceFromCameraToLandmark = self.landMarkSize / (2 * math.tan(angularSize / 2))
        transform = self.motionProxy.getTransform(currentCamera, 2, True)
        transformList = almath.vectorFloat(transform)
        robotToCamera = almath.Transform(transformList)
        cameraToLandmarkRotationTransform = almath.Transform_from3DRotation(0, wyCamera, wzCamera)
        cameraToLandmarkTranslationTransform = almath.Transform(distanceFromCameraToLandmark, 0, 0)
        robotToLandmark = robotToCamera * cameraToLandmarkRotationTransform * cameraToLandmarkTranslationTransform
        landMarkX = robotToLandmark.r1_c4
        landMarkY = robotToLandmark.r2_c4
        self.landmarkProxy.unsubscribe("landmark")
        yawAngle = math.atan2(landMarkY, landMarkX)
        self.disX = landMarkX
        self.disY = landMarkY
        self.dist = distanceFromCameraToLandmark
        self.yawAngle = yawAngle


if __name__ == '__main__':
    # stick_detect = StickDetection("127.0.0.1")
    # with codecs.open("timeInfo.txt", 'a', encoding='utf-8') as f:
    #    f.write("\n")

    stick_detect = StickDetection("192.168.137.117")
    for ii in range(20):
        s_time = time.time()
        stick_detect.updateStickData()
        stick_img = stick_detect.stickImg
        with codecs.open("timeInfo.txt", 'a', encoding='utf-8') as f:
            f.write("all time: {:.2}s\n".format(time.time() - s_time))
        if stick_img is not None:
            stick_detect.showStickPosition()
            # cv2.imshow("stick", stick_img)
            # cv2.waitKey(1000)
            # cv2.destroyWindow("stick")
    '''
    ball_detect = RedBallDetection("192.168.137.117")
    for ii in range(20):
        s_time = time.time()
        ball_detect.updateBallData()
        ball_img = ball_detect.ballImg
        with codecs.open("timeInfo.txt", 'a', encoding='utf-8') as f:
            f.write("all time: {:.2}s\n".format(time.time() - s_time))
        if ball_img is not None:
            ball_detect.showBallPosition()
            # cv2.imshow("ball", ball_img)
            # cv2.waitKey(1000)
            # cv2.destroyWindow("ball")
        else:
            print("no ball!")
    '''
# 169.254.189.102
