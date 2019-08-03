from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import numpy as np
import vision_definitions as vd
import time
from ConfigureNao import ConfigureNao
from naoqi import ALProxy
import motion
import math
import almath
import cv2


class VisualBasis(ConfigureNao):
    def __init__(self, IP, PORT, cameraId=vd.kBottomCamera, resolution=vd.kVGA):
        super(VisualBasis, self).__init__(IP, PORT)
        self.cameraId = cameraId
        self.cameraName = "CameraBottom" if self.cameraId == vd.kBottomCamera else "CameraTop"
        self.resolution = resolution
        self.colorSpace = vd.kBGRColorSpace
        self.fps = 20
        self.frameHeight = 0
        self.frameWidth = 0
        self.frameChannels = 0
        self.frameArray = None
        self.cameraPitchRange = 47.64 * np.pi / 180
        self.cameraYawRange = 60.92 * np.pi / 180
        self.cameraProxy.setActiveCamera(self.cameraId)

    def updateFrame(self, client="python_client"):
        if self.cameraProxy.getActiveCamera() != self.cameraId:
            self.cameraProxy.setActiveCamera(self.cameraId)
            time.sleep(1)

        videoClient = self.cameraProxy.subscribe(client, self.resolution, self.colorSpace, self.fps)
        frame = self.cameraProxy.getImageRemote(videoClient)
        self.cameraProxy.unsubscribe(videoClient)
        try:
            self.frameWidth = frame[0]
            self.frameHeight = frame[1]
            self.frameChannels = frame[2]
            self.frameArray = np.frombuffer(frame[6], dtype=np.uint8).reshape([frame[1], frame[0], frame[2]])
        except IndexError:
            print("get image failed!")

    def getFrameArray(self):
        if self.frameArray is None:
            return np.array([])
        return self.frameArray

    def showFrame(self):
        if self.frameArray is None:
            print("please get an image from Nao with the method updateFrame()")
        else:
            cv.imshow("current frame", self.frameArray)

    def printFrameData(self):
        print("frame height = ", self.frameHeight)
        print("frame width = ", self.frameWidth)
        print("frame channels = ", self.frameChannels)
        print("frame shape = ", self.frameArray.shape)

    def saveFrame(self, framePath):
        cv.imwrite(framePath, self.frameArray)
        print("current frame image has been saved in", framePath)

    def setParam(self, paramName=None, paramValue=None):
        raise NotImplementedError


class BallDetect(VisualBasis):
    def __init__(self, IP, PORT=9559, cameraId=vd.kBottomCamera, resolution=vd.kVGA, writeFrame=False):
        super(BallDetect, self).__init__(IP, PORT, cameraId, resolution)
        self.ballData = {"centerX": 0, "centerY": 0, "radius": 0}
        self.ballPosition = {"disX": 0, "disY": 0, "angle": 0, "stickX": 0, "stickY": 0, "HitX": 0, "HitY": 0}
        self.ballRadius = 0.021
        self.writeFrame = writeFrame

    def __getChannelAndBlur(self, color):
        try:
            channelB = self.frameArray[:, :, 0]
            channelG = self.frameArray[:, :, 1]
            channelR = self.frameArray[:, :, 2]
        except:
            print("no image detected!")
        Hm = 6
        if color == "red":
            channelB = channelB * 0.1 * Hm
            channelG = channelG * 0.1 * Hm
            channelR = channelR - channelB - channelG
            channelR = 3 * channelR
            channelR = cv2.GaussianBlur(channelR, (9, 9), 1.5)
            channelR[channelR < 0] = 0
            channelR[channelR > 255] = 255
            return np.uint8(np.round(channelR))
        elif color == "blue":
            channelR = channelR * 0.1 * Hm
            channelG = channelG * 0.1 * Hm
            channelB = channelB - channelG - channelR
            channelB = cv2.GaussianBlur(channelB, (9, 9), 1.5)
            channelB[channelB < 0] = 0
            channelB[channelB > 255] = 255
            return np.uint8(np.round(channelB))
        elif color == "green":
            channelB = channelB * 0.1 * Hm
            channelR = channelR * 0.1 * Hm
            channelG = channelG - channelB - channelR
            channelG = 3 * channelG
            channelG[channelG < 0] = 0
            channelG[channelG > 255] = 255
            return np.uint8(np.round(channelG))
        else:
            print("can't not recognize the color!")
            print("supported color:red,green and blue.")
            return None

    def __binImageHSV(self, minHSV1, maxHSV1, minHSV2, maxHSV2):
        try:
            frameArray = self.frameArray.copy()
            imgHSV = cv2.cvtColor(frameArray, cv2.COLOR_BGR2HSV)
        except:
            print("no image detected!")
        else:
            frameBin1 = cv2.inRange(imgHSV, minHSV1, maxHSV1)
            frameBin2 = cv2.inRange(imgHSV, minHSV2, maxHSV2)
            frameBin = np.maximum(frameBin1, frameBin2)
            frameBin = cv2.GaussianBlur(frameBin, (9, 9), 1.5)
            return frameBin

    def __findCircles(self, img, minDist, minRadius, maxRadius):
        gradient_name = cv2.HOUGH_GRADIENT
        circles = cv2.HoughCircles(np.uint8(img), gradient_name, 1, \
                                   minDist, param1=150, param2=15, \
                                   minRadius=minRadius, maxRadius=maxRadius)
        if circles is None:
            return np.uint16([])
        else:
            return np.uint16(np.around(circles[0,]))

    def __selectCircle(self, circles):
        if circles.shape[0] == 0:
            return circles
        if circles.shape[0] == 1:
            centerX = circles[0][0]
            centerY = circles[0][1]
            radius = circles[0][2]
            initX = centerX - 2 * radius
            initY = centerY - 2 * radius
            if (initX < 0 or initY < 0 or (initX + 4 * radius) > self.frameWidth or \
                    (initY + 4 * radius) > self.frameHeight or radius < 1):
                return circles
        channelB = self.frameArray[:, :, 0]
        channelG = self.frameArray[:, :, 1]
        channelR = self.frameArray[:, :, 2]
        rRatioMin = 1.0
        circleSelected = np.uint16([])
        for circle in circles:
            centerX = circle[0]
            centerY = circle[1]
            radius = circle[2]
            initX = centerX - 2 * radius
            initY = centerY - 2 * radius
            if initX < 0 or initY < 0 or (initX + 4 * radius) > self.frameWidth or \
                    (initY + 4 * radius) > self.frameHeight or radius < 1:
                continue
            rectBallArea = self.frameArray[initY:initY + 4 * radius + 1, initX:initX + 4 * radius + 1, :]
            bFlat = np.float16(rectBallArea[:, :, 0].flatten())
            gFlat = np.float16(rectBallArea[:, :, 1].flatten())
            rFlat = np.float16(rectBallArea[:, :, 2].flatten())
            rScore1 = np.uint8(rFlat > 1.0 * gFlat)
            rScore2 = np.uint8(rFlat > 1.0 * bFlat)
            rScore = float(np.sum(rScore1 * rScore2))
            gScore = float(np.sum(np.uint8(gFlat > 1.0 * rFlat)))
            rRatio = rScore / len(rFlat)
            gRatio = gScore / len(gFlat)
            if rRatio >= 0.12 and gRatio >= 0.1 and abs(rRatio - 0.19) < abs(rRatioMin - 0.19):
                circleSelected = circle
                rRatioMin = rRatio
        return circleSelected

    def __updateBallPositionFitting(self, standState):
        bottomCameraDirection = {"standInit": 49.2, "standUp": 39.7}
        ballRadius = self.ballRadius
        try:
            cameraDirection = bottomCameraDirection[standState]
        except KeyError:
            print("Error! unknown standState,please check the value of stand state!")
        else:
            if self.ballData["radius"] == 0:
                self.ballPosition = {"disX": 0, "disY": 0, "angle": 0, "stickX": 0, "stickY": 0, "HitX": 0, "HitY": 0}
            else:
                centerX = self.ballData["centerX"]
                centerY = self.ballData["centerY"]
                radius = self.ballData["radius"]
                cameraPosition = self.motionProxy.getPosition("CameraBottom", 2, True)
                cameraX = cameraPosition[0]
                cameraY = cameraPosition[1]
                cameraHeight = cameraPosition[2]
                headPitches = self.motionProxy.getAngles("HeadPitch", True)
                headPitch = headPitches[0]
                headYaws = self.motionProxy.getAngles("HeadYaw", True)
                headYaw = headYaws[0]
                imgCenterX = self.frameWidth / 2
                imgCenterY = self.frameHeight / 2
                ballPitch = (centerY - imgCenterY) * self.cameraPitchRange / self.frameHeight
                ballYaw = (imgCenterX - centerX) * self.cameraYawRange / self.frameWidth
                dPitch = (cameraHeight - ballRadius) / np.tan(cameraDirection / 180 * np.pi + headPitch + ballPitch)
                dYaw = dPitch / np.cos(ballYaw)
                ballX = dYaw * np.cos(ballYaw + headYaw) + cameraX
                ballY = dYaw * np.sin(ballYaw + headYaw) + cameraY
                ballYaw = np.arctan2(ballY, ballX)
                self.ballPosition["disX"] = ballX
                if (standState == "standInit"):
                    ky = 45.513 * ballX ** 4 - 109.66 * ballX ** 3 + 104.2 * ballX ** 2 - 44.218 * ballX + 8.5526
                    ballY = ky * ballY
                    ballYaw = np.arctan2(ballY, ballX)
                self.ballPosition["disY"] = ballY
            self.ballPosition["angle"] = ballYaw

    """
    def computeHitBallPosition(self, HitBallD=0.15, stickHeight=0.47, stickWidth=0.047):
        :param robotCameraH: robotCamera Height
        :param sizeX: robotCamera current sizeX
        :param sizeY: robotCamera current sizeY
        :param imgWidth: img Width
        :param imgHeight: img height
        :param sCenterX: stick centX in img
        :param sCenterY: stick centY in img
        :param ballX: ballX
        :param ballY: ballY
        :param HitBallD: The distance from the hitting point to the ball, meters
        :param stickHeight: stick height, meters
        :return: A dict, stickX, stickY, HitX, HitY
        stickdetected = StickDetect()
        stickCenter = stickdetected.updateStickData()
        if stickCenter != []:
            rat_stick = stickHeight / stickWidth
            sCenterX = stickCenter[0] + stickCenter[2] / 2
            sCenterY = stickCenter[1] + rat_stick * stickCenter[2] / 2
            ratZ = 47.67 * math.pi / 180
            ratX = 60.97 * math.pi / 180
            cameraPosition = self.motionProxy.getPosition("CameraBottom", 2, True)
            hb = HitBallD  # The distance from the hitting point to the ball
            rh = cameraPosition[2]  # robotCamera Height
            sh = stickHeight  # stick height
            sizeX = cameraPosition[0]  # robotCamera current sizeX
            sizeY = cameraPosition[1]  # robotCamera current sizeY
            imgW = self.frameWidth  # img Width
            imgH = self.frameHeight  # img height
            # compute robot-stick distance
            d = math.pi / 2 * (rh - sh / 2) / (math.pi / 2 - (sizeY + (sCenterY - imgH / 2) * ratZ / imgH))
            stickAngle = sizeX + (imgW / 2 - sCenterX) * ratX / imgW
            stickX = d * math.sin(stickAngle)
            stickY = d * math.cos(stickAngle)
            ballX = self.ballPosition["disX"]
            ballY = self.ballPosition["disY"]

            k = (stickY - ballY) / (stickX - ballX)

            HitX = math.sqrt(math.sqrt(hb) / (k + 1)) + ballX  # Hit ball position
            HitY = k * (math.sqrt(math.sqrt(hb) / (k + 1))) + ballY
            self.ballPosition["stickX"] = stickX
            self.ballPosition["stickY"] = stickY
            self.ballPosition["HitX"] = HitX
            self.ballPosition["HitY"] = HitY
        """

    def __writeFrame(self, saveDir="./ballData"):
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        saveName = str(int(time.time()))
        saveImgPath = os.path.join(saveDir, saveName + ".jpg")
        try:
            cv2.imwrite(saveImgPath, self.frameArray)
        except:
            print("Error when saveing current frame!")

    def updateBallData(self, client="python_client", standState="standInit", color="red",
                       colorSpace="BGR", minHSV1=np.array([0, 43, 46]),
                       maxHSV1=np.array([10, 255, 255]), minHSV2=np.array([156, 43, 46]),
                       maxHSV2=np.array([180, 255, 255]), saveFrameBin=False):
        self.updateFrame(client)
        minDist = int(self.frameHeight / 30.0)
        minRadius = 1
        maxRadius = int(self.frameHeight / 10.0)
        if colorSpace == "BGR":
            grayFrame = self.__getChannelAndBlur(color)
        else:
            grayFrame = self.__binImageHSV(minHSV1, maxHSV1, minHSV2, maxHSV2)
        if saveFrameBin:
            self._frameBin = grayFrame.copy()
        circles = self.__findCircles(grayFrame, minDist, minRadius, maxRadius)
        circle = self.__selectCircle(circles)
        if circle.shape[0] == 0:
            self.ballData = {"centerX": 0, "centerY": 0, "radius": 0}
            self.ballPosition = {"disX": 0, "disY": 0, "angle": 0, "stickX": 0, "stickY": 0, "HitX": 0, "HitY": 0}
            return {}
        else:
            circle = circle.reshape([-1, 3])
            self.ballData = {"centerX": circle[0][0], "centerY": circle[0][1], "radius": circle[0][2]}
            self.__updateBallPositionFitting(standState=standState)
            print("ballX: ", self.ballPosition["disX"])
            print("ballY: ", self.ballPosition["disY"])
            print("ballYaw: ", self.ballPosition["angle"])
            # self.computeHitBallPosition()
            return self.ballPosition
        if self.writeFrame == True:
            self.__writeFrame()

    def getBallPosition(self):
        disX = self.ballPosition["disX"]
        disY = self.ballPosition["disY"]
        angle = self.ballPosition["angle"]
        return [disX, disY, angle]

    def getBallInfoInImage(self):
        centerX = self.ballData["centerX"]
        centerY = self.ballData["centerY"]
        radius = self.ballData["radius"]
        return [centerX, centerY, radius]

    def showBallPosition(self):
        if self.ballData["radius"] == 0:
            print("ball position=", (self.ballPosition["disX"], self.ballPosition["disY"]))
            cv2.imshow("ball position", self.frameArray)
        else:
            print("ballX= ", self.ballData["centerX"])
            print("ballY= ", self.ballData["centerY"])
            print("ball position= ", (self.ballPosition["disX"], self.ballPosition["disY"]))
            print("ball direction=", self.ballPosition["angle"] * 180 / 3.14)
            print("stickX position= ", self.ballPosition["stickX"])
            print("stickY position=", self.ballPosition["stickY"])
            print("ball HitX=", self.ballPosition["HitX"])
            print("ball HitY=", self.ballPosition["HitY"])
            frameArray = self.frameArray.copy()
            cv2.circle(frameArray, (self.ballData["centerX"], self.ballData["centerY"]),
                       self.ballData["radius"], (250, 150, 150), 2)
            cv2.circle(frameArray, (self.ballData["centerX"], self.ballData["centerY"]),
                       2, (50, 250, 50), 3)
            cv2.imshow("ball position", frameArray)

    def sliderHSV(self, client):
        def __nothing():
            pass

        windowName = "slider for ball detection"
        cv2.namedWindow(windowName)
        cv2.createTrackbar("minS1", windowName, 43, 60, __nothing)
        cv2.createTrackbar("minV1", windowName, 46, 65, __nothing)
        cv2.createTrackbar("maxH1", windowName, 10, 20, __nothing)
        cv2.createTrackbar("minH2", windowName, 156, 175, __nothing)
        while 1:
            time.sleep(0.2)
            self.updateFrame(client)
            minS1 = cv2.getTrackbarPos("minS1", windowName)
            minV1 = cv2.getTrackbarPos("minV1", windowName)
            maxH1 = cv2.getTrackbarPos("maxH1", windowName)
            minH2 = cv2.getTrackbarPos("minH2", windowName)
            minHSV1 = np.array([0, minS1, minV1])
            maxHSV1 = np.array([maxH1, 255, 255])
            minHSV2 = np.array([minH2, minS1, minV1])
            maxHSV2 = np.array([180, 255, 255])
            self.updateBallData(client, colorSpace="HSV", standState="standUp", minHSV1=minHSV1, maxHSV1=maxHSV1,
                                minHSV2=minHSV2,
                                maxHSV2=maxHSV2, saveFrameBin=True)
            self.updateBallData(client, colorSpace="HSV", standState="standUp", saveFrameBin=True)
            cv2.imshow(windowName, self._frameBin)
            self.showBallPosition()
            k = cv2.waitKey(10) & 0xFf
            if k == 27:
                break

        cv2.destroyAllWindows()


class StickDetect(VisualBasis):
    def __init__(self, IP="192.168.1.104", PORT=9559, cameraId=vd.kTopCamera, resolution=vd.kVGA, writeFrame=False):
        super(StickDetect, self).__init__(IP, PORT, cameraId, resolution)
        self.boundRect = []
        self.cropKeep = 1
        self.stickAngle = 0.0
        self.writeFrame = writeFrame

    def __preprocess(self, minHSV, maxHSV, cropKeep, morphology):
        self.cropKeep = cropKeep
        frameArray = self.frameArray
        height = self.frameHeight
        width = self.frameWidth
        try:
            frameArray = frameArray[int((1 - cropKeep) * height):, :]
        except IndexError:
            print("error happened when crop the image!")
        frameHSV = cv2.cvtColor(frameArray, cv2.COLOR_BGR2HSV)
        frameBin = cv2.inRange(frameHSV, minHSV, maxHSV)
        kernelErosion = np.ones((5, 5), np.uint8)
        kerneDilation = np.ones((5, 5), np.uint8)
        frameBin = cv2.erode(frameBin, kernelErosion, iterations=1)
        frameBin = cv2.dilate(frameBin, kerneDilation, iterations=1)
        frameBin = cv2.GaussianBlur(frameBin, (9, 9), 0)
        return frameBin

    def __findStick(self, frameBin, minPerimeter, minArea):
        rects = []
        contours, _ = cv2.findContours(frameBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return rects
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            if perimeter > minPerimeter and area > minArea:
                x, y, w, h = cv2.cv2.boundingRect(contour)
                rects.append([x, y, w, h])
        if len(rects) == 0:
            return rects
        rects = [rect for rect in rects if (1.0 * rect[3] / rect[2]) > 0.8]
        if len(rects) == 0:
            return rects
        rects = np.array(rects)
        rect = rects[np.argmax(1.0 * (rects[:, -1]) / rects[:, -2]),]
        rect[1] += int(self.frameHeight * (1 - self.cropKeep))
        return rect

    def __writeFrame(self, saveDir="./stickData"):
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        saveName = str(int(time.time()))
        saveImgPath = os.path.join(saveDir, saveName + ".jpg")
        try:
            cv2.imwrite(saveImgPath, self.frameArray)
        except:
            print("Error when saveing current frame!")

    def updateStickData(self, client="test", minHSV=np.array([27, 55, 115]),
                        maxHSV=np.array([45, 255, 255]), cropKeep=0.75,
                        morphology=True, savePreprocessImg=False):
        self.updateFrame(client)
        minPerimeter = self.frameHeight / 8.0
        minArea = self.frameHeight * self.frameWidth / 1000.0
        frameBin = self.__preprocess(minHSV, maxHSV, cropKeep, morphology)
        centerX = 0
        centerY = 0
        if savePreprocessImg:
            self.frameBin = frameBin.copy()
        rect = self.__findStick(frameBin, minPerimeter, minArea)
        if rect == []:
            self.boundRect = []
            self.stickAngle = 0.0
        else:
            self.boundRect = rect
            centerX = rect[0] + rect[2] / 2
            centerY = rect[1] + rect[3] / 2
            width = self.frameWidth * 1.0
            self.stickAngle = (width / 2 - centerX) / width * self.cameraYawRange
            cameraPosition = self.motionProxy.getPosition("Head", 2, True)
            cameraY = cameraPosition[5]
            self.stickAngle += cameraY
            if self.writeFrame == True:
                self.__writeFrame()
            return rect
        return []

    def showStickPosition(self):
        if self.boundRect == []:
            cv2.imshow("stick Postion", self.frameArray)
        else:
            [x, y, w, h] = self.boundRect
            frame = self.frameArray.copy()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow("stick position", frame)

    def slider(self, client):
        def __nothing():
            pass

        windowName = "slider for stick detection"
        cv2.namedWindow(windowName)
        cv2.createTrackbar("minH", windowName, 27, 45, __nothing)
        cv2.createTrackbar("minS", windowName, 55, 75, __nothing)
        cv2.createTrackbar("minV", windowName, 115, 150, __nothing)
        cv2.createTrackbar("maxH", windowName, 45, 70, __nothing)
        while 1:
            self.updateFrame(client)
            minH = cv2.getTrackbarPos("minH", windowName)
            minS = cv2.getTrackbarPos("minS", windowName)
            minV = cv2.getTrackbarPos("minV", windowName)
            maxH = cv2.getTrackbarPos("maxH", windowName)
            minHSV = np.array([minH, minS, minV])
            maxHSV = np.array([maxH, 255, 255])
            self.updateStickData(client, minHSV, maxHSV, savePreprocessImg=True)
            cv2.imshow(windowName, self._frameBin)
            self.showStickPosition()
            k = cv2.waitKey(10) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()


class LandMarkDetect(ConfigureNao):
    def __init__(self, IP, PORT=9559, cameraId=vd.kTopCamera, landMarkSize=0.105):
        super(LandMarkDetect, self).__init__(IP, PORT)
        self.cameraId = cameraId
        self.cameraName = "CameraTop" if cameraId == vd.kTopCamera else "CameraBottom"
        self.landMarkSize = landMarkSize
        self.disX = 0
        self.disY = 0
        self.dist = 0
        self.yawAngle = 0
        self.cameraProxy.setActiveCamera(self.cameraId)

    def updateLandMarkData(self, client="landMark"):
        if self.cameraProxy.getActiveCamera() != self.cameraId:
            self.cameraProxy.setActiveCamera(self.cameraId)
            time.sleep(1)
        self.landMarkProxy.subscribe(client)
        markData = self.memoryProxy.getData("LandmarkDetected")
        self.cameraProxy.unsubscribe(client)
        if (markData is None or len(markData) == 0):
            self.disX = 0
            self.disY = 0
            self.dist = 0
            self.yawAngle = 0
            return []

        else:
            wzCamera = markData[1][0][0][1]
            wyCamera = markData[1][0][0][2]
            angularSize = markData[1][0][0][3]
            distCameraToLandmark = self.landMarkSize / (2 * math.tan(angularSize / 2))
            transform = self.motionProxy.getTransform(self.cameraName, 2, True)
            transformList = almath.vectorFloat(transform)
            robotToCamera = almath.Transform(transformList)
            cameraToLandmarkRotTrans = almath.Transform_from3DRotation(0, wyCamera, wzCamera)
            cameraToLandmarkTranslationTrans = almath.Transform(distCameraToLandmark, 0, 0)
            robotToLandmark = robotToCamera * \
                              cameraToLandmarkRotTrans * \
                              cameraToLandmarkTranslationTrans
            self.disX = robotToLandmark.r1_c4
            self.disY = robotToLandmark.r2_c4
            self.dist = np.sqrt(self.disX ** 2 + self.disY ** 2)
            self.yawAngle = math.atan2(self.disY, self.disX)
            return [self.disX, self.disY, self.dist, self.yawAngle]

    def getLandMarkData(self):
        return [self.disX, self.disY, self.dist, self.yawAngle]

    def showLandMarkData(self):
        print("disX=", self.disX)
        print("disY=", self.disY)
        print("dis =", self.dist)
        print("yaw angle = ", self.yawAngle * 180.0 / np.pi)
