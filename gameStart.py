from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from naoqi import ALProxy

import numpy as np
import almath
import time
import math
import cv2
import sys
import os
import threading

from ConfigureNao import *
from visualTask import *
import vision_definitions as vd
from stepStatus import *

rad = almath.TO_RAD


class GolfGame(ConfigureNao):
    def __init__(self, robotIP, port):
        super(GolfGame, self).__init__(robotIP, port)
        self.robotIP = robotIP
        self.port = port
        self.ball_info = [0, 0, 0]
        self.landmark_info = [0, 0, 0, 0]
        self.stickAngle = 0.01
        self.pitchAngle = -10
        self.yawAngle = 0
        self.hitballtimes = 0
        self.flag = True

        self.firstHitSpeed_1_blue = 0.57
        self.firstHitSpeed_1_red = 0.61
        self.firstHitSpeed_2_blue = 0.64
        self.firstHitSpeed_2_red = 0.53
        self.firstHitSpeed_3_blue = 0.412
        self.firstHitSpeed_3_red = 0.38

        self.ballDetect = BallDetect(robotIP, cameraId=vd.kBottomCamera)
        self.stickDetect = StickDetect(robotIP, cameraId=vd.kTopCamera)
        self.landmarkDetect = LandMarkDetect(robotIP, cameraId=vd.kTopCamera)
        self.walkconfiguration = WalkConfiguration()
        time.sleep(10)

    def gameStart(self):
        while True:
            FrontFlag = self.memoryProxy.getData("Device/SubDeviceList/Head/Touch/Front/Sensor/Value")
            MiddleFlag = self.memoryProxy.getData("Device/SubDeviceList/Head/Touch/Middle/Sensor/Value")
            RearFlag = self.memoryProxy.getData("Device/SubDeviceList/Head/Touch/Rear/Sensor/Value")
            if FrontFlag == 1:
                self.prepare()
                self.gameTask_1()
            if MiddleFlag == 1:
                self.prepare()
                self.gameTask_2()
            if RearFlag == 1:
                self.prepare()
                self.gameTask_3()

    def prepare(self):
        self.motionProxy.wakeUp()
        self.postureProxy.goToPosture("Standing", 0.5)
        self.motionProxy.setStiffnesses("Body", 1.0)

    def findball(self, pitchAngle=-10, yawAngle=0, isShowFlag=False):
        Names = ["HeadPitch", "HeadYaw"]
        Angles = [pitchAngle * rad, yawAngle * rad]
        self.pitchAngle = pitchAngle
        self.yawAngle = yawAngle
        self.motionProxy.angleInterpolationWithSpeed(Names, Angles, 0.1)
        time.sleep(0.5)

        self.ballDetect.updateBallData(client="findball55", colorSpace="HSV", color="red", fitting=True)
        self.ball_info = self.ballDetect.getBallPosition()
        if isShowFlag is True:
            print("picture show:")
            self.ballDetect.showBallPosition()
            cv2.waitKey(2)

        if self.ball_info[0] > 0:
            self.tts.say("find ball")
            return True
        else:
            self.tts.say("no ball")
            return False

    def findStick(self, pitchAngle=-10, yawAngle=0, isShowFlag=False):
        Names = ["HeadPitch", "HeadYaw"]
        Angles = [pitchAngle * rad, yawAngle * rad]
        self.motionProxy.angleInterpolationWithSpeed(Names, Angles, 0.1)
        time.sleep(0.5)

        self.stickDetect.updateStickData(minHSV=np.array([32, 75, 7]), maxHSV=np.array([40, 255, 255]))
        self.stickAngle = self.stickDetect.stickAngle
        if isShowFlag:
            self.stickDetect.showStickPosition()
            cv2.waitKey(2)

        if self.stickAngle > 0.002 or self.stickAngle < -0.002:
            self.tts.say("find stick")
            print("find stick:", self.stickAngle * almath.TO_DEG)
            return True
        else:
            self.tts.say("no stick")
            return False

    def findStcik2(self, pitchAngle=-10, yawAngle=0, isShowFlag=False):
        Names = ["HeadPitch", "HeadYaw"]
        Angles = [pitchAngle * rad, yawAngle * rad]
        self.motionProxy.angleInterpolationWithSpeed(Names, Angles, 0.1)
        time.sleep(0.5)

        self.stickDetect.updateStickData(minHSV=np.array([27, 55, 84]), maxHSV=np.array([43, 255, 255]))
        self.stickAngle = self.stickDetect.stickAngle
        if isShowFlag:
            self.stickDetect.showStickPosition()
            cv2.waitKey(2)

        if self.stickAngle > 0.002 or self.stickAngle < -0.002:
            self.tts.say("find stick")
            print("find stick:", self.stickAngle * almath.TO_DEG)
            return True
        else:
            self.tts.say("no stick")
            return False

    def findLandmark(self, pitchAngle=-10, yawAngle=0):
        Names = ["HeadPitch", "HeadYaw"]
        Angles = [pitchAngle * rad, yawAngle * rad]
        self.motionProxy.angleInterpolationWithSpeed(Names, Angles, 0.1)
        time.sleep(0.5)

        self.landmarkDetect.updateLandMarkData()
        self.landmark_info = self.landmarkDetect.getLandMarkData()

        if abs(self.landmark_info[3]) > 0.0001:
            self.tts.say("find mark")
            print("find mark:", self.landmark_info[3] * almath.TO_DEG)
            print("find mark all", self.landmark_info)
            return True
        else:
            self.tts.say("no mark")
            return False

    def forehandToHitball(self, hitSpeed):
        self.AddressingTheBall()
        self.Batting(hitSpeed)
        self.ReceivingPole()

    def backhandToHitball(self, hitSpeed):
        self.AddressingTheBall2()
        self.Batting2(hitSpeed)
        self.ReceivingPole()

    def moveheadToFindball(self, pitchAngles=[-10], yawAngles=[-40, -15, 15, 40]):
        isfindBall = False
        for pitchAngle in pitchAngles:
            for yawAngle in yawAngles:
                isfindBall = self.findball(pitchAngle=pitchAngle, yawAngle=yawAngle)
                if isfindBall:
                    break
            if isfindBall:
                break
        self.motionProxy.angleInterpolationWithSpeed("HeadYaw", 0, 0.1)
        return isfindBall

    def moveheadToFindstick(self, yawAngles):
        isfindStick = False
        for yawAngle in yawAngles:
            isfindStick = self.findStick(pitchAngle=-10, yawAngle=yawAngle)
            if isfindStick:
                break
        return isfindStick

    def moveheadTofindstick2(self, yawAngles):
        isfindStick = False
        for yawAngle in yawAngles:
            isfindStick = self.findStick(pitchAngle=-10, yawAngle=yawAngle)
            if isfindStick:
                break
        return isfindStick

    def moveheadToFindLandmark(self, yawAngles):
        isfindLandmark = False
        for yawAngle in yawAngles:
            isfindLandmark = self.findLandmark(pitchAngle=-10, yawAngle=yawAngle)
            if isfindLandmark:
                break
        return isfindLandmark

    def moveheadToFindLandmarkandStick(self, yawAngles):
        targetDis = 0
        targetAngle = 0
        compensateAngle = 0
        for yawAngle in yawAngles:
            isfindLandmark = self.findLandmark(pitchAngle=0, yawAngle=yawAngle)
            if isfindLandmark:
                targetDis = self.landmark_info[2]
                targetAngle = self.landmark_info[3]
                cmpensateAngle = 10 * rad
                break
            isfindStick = self.findStick(pitchAngle=0, yawAngle=yawAngle)
            if isfindStick:
                targetDis = -999
                targetAngle = self.stickAngle
                compensateAngle = self.compensateAngle1(self.hitballtimes)
                break
        print("compensateAngle", compensateAngle)
        return [targetDis, targetAngle, compensateAngle]

    def walkToBall(self, min_ball_d=0.4, max_ball_theta=0.15):
        isfindBall = self.findball()
        degree = 0
        if isfindBall is False:
            while True:
                isfindBall = self.moveheadToFindball(pitchAngles=[-20, -15, 0, 10, 20])
                if isfindBall is False:
                    self.motionProxy.setMoveArmsEnabled(False, False)
                    self.motionProxy.moveTo(0.5, 0, 0, self.walkconfiguration.WalkCircleLittle_blue())
                else:
                    break
        self.motionProxy.moveTo(0, 0, self.ball_info[2], self.walkconfiguration.WalkLineLittle_blue())
        self.motionProxy.angleInterpolationWithSpeed("HeadYaw", 0, 0.1)

        moveTask_1 = self.motionProxy.post.moveTo(1.5 * self.ball_info[0], 0, 0,
                                                  self.walkconfiguration.WalkLineLittle_blue())
        while True:
            isfindBall = self.findball(self.pitchAngle, 0)
            if isfindBall is False:
                self.motionProxy.stop(moveTask_1)
                time.sleep(0.5)
                isfindBall_2 = self.moveheadToFindball(
                    pitchAngles=[self.pitchAngle, self.pitchAngle + 10, self.pitchAngle + 20, self.pitchAngle - 10],
                    yawAngles=[-15, 15])
                if isfindBall_2:
                    moveTask_1 = self.motionProxy.post.moveTo(1.5 * self.ball_info[0], 0, 0,
                                                              self.walkconfiguration.WalkLineLittle_blue())

            ball_d = ((self.ball_info[0] ** 2 + self.ball_info[1] ** 2) ** 0.5)
            ball_theta = abs(self.ball_info[2])
            time.sleep(0.5)

            Names = ["HeadPitch"]
            if ball_d > 0.6:
                self.pitchAngle = 0
            elif 0.4 < ball_d < 0.6:
                self.pitchAngle = 10
            elif 0.15 < ball_d < 0.4:
                self.pitchAngle = 20
            self.motionProxy.angleInterpolationWithSpeed(Names, self.pitchAngle * rad, 0.1)

            if min_ball_d / 3 < self.ball_info[0] < 0.4:
                self.tts.say("I am right")
                self.motionProxy.stop(moveTask_1)
                time.sleep(0.5)
                break

            if 0.13 < ball_d < min_ball_d:
                self.tts.say("I am right.")
                self.motionProxy.stop(moveTask_1)
                time.sleep(0.5)
                break
            elif 0.22 < abs(ball_theta) < max_ball_theta:
                if min_ball_d / 3 < ball_d < min_ball_d:
                    self.tts.say("I am right.")
                    self.motionProxy.stop(moveTask_1)
                    time.sleep(0.5)
                    break
                else:
                    continue
            elif abs(ball_theta) > max_ball_theta:
                self.tts.say("I am wrong.")
                self.motionProxy.stop(moveTask_1)
                time.sleep(0.5)
                self.motionProxy.moveTo(0, 0, self.ball_info[2], self.walkconfiguration.WalkLineMiddle_blue())
                moveTask_1 = self.motionProxy.post.moveTo(1 * self.ball_info[0], 0, 0,
                                                          self.walkconfiguration.WalkLineMiddle_blue())

    def walkToBall2(self, min_ball_d=0.4, max_ball_theta=0.15):
        isfindBall = self.findball()
        if isfindBall is False:
            while True:
                isfindBall = self.moveheadToFindball(pitchAngles=[0, -15, 15])
                if isfindBall is False:
                    self.motionProxy.setMoveArmsEnabled(False, False)
                    self.motionProxy.moveTo(0.5, 0, 0, self.walkconfiguration.WalkLineLittle_blue())
                else:
                    break
        self.motionProxy.moveTo(0, 0, self.ball_info[2], self.walkconfiguration.WalkCircleLittle_blue())
        self.motionProxy.angleInterpolationWithSpeed("HeadYaw", 0, 0.1)

        moveTask_1 = self.motionProxy.post.moveTo(1.5 * self.ball_info[0], 0, 0,
                                                  self.walkconfiguration.WalkLineLittle_blue())
        while True:
            isfindBall = self.findball(self.pitchAngle, 0)
            if isfindBall is False:
                self.motionProxy.stop(moveTask_1)
                time.sleep(0.5)
                isfindBall_2 = self.moveheadToFindball(
                    pitchAngles=[self.pitchAngle, self.pitchAngle + 10, self.pitchAngle + 20, self.pitchAngle - 10],
                    yawAngles=[-15, 15])
                if isfindBall_2:
                    moveTask_1 = self.motionProxy.post.moveTo(1.5 * self.ball_info[0], 0, 0,
                                                              self.walkconfiguration.WalkLineLittle_blue())

            ball_d = ((self.ball_info[0] ** 2 + self.ball_info[1] ** 2) ** 0.5)
            ball_theta = abs(self.ball_info[2])
            time.sleep(0.5)

            Names = ["HeadPitch"]
            if ball_d > 0.6:
                self.pitchAngle = 0
            elif 0.4 < ball_d < 0.6:
                self.pitchAngle = 10
            elif 0.15 < ball_d < 0.4:
                self.pitchAngle = 20
            self.motionProxy.angleInterpolationWithSpeed(Names, self.pitchAngle * rad, 0.1)

            if min_ball_d / 3 < self.ball_info[0] < 0.4:
                self.tts.say("I am right")
                self.motionProxy.stop(moveTask_1)
                time.sleep(0.5)
                break

            if 0.13 < ball_d < min_ball_d:
                self.tts.say("I am right.")
                self.motionProxy.stop(moveTask_1)
                time.sleep(0.5)
                break
            elif 0.22 < abs(ball_theta) < max_ball_theta:
                if min_ball_d / 3 < ball_d < min_ball_d:
                    self.tts.say("I am right.")
                    self.motionProxy.stop(moveTask_1)
                    time.sleep(0.5)
                    break
                else:
                    continue
            elif abs(ball_theta) > max_ball_theta:
                self.tts.say("I am wrong.")
                self.motionProxy.stop(moveTask_1)
                time.sleep(0.5)
                self.motionProxy.moveTo(0, 0, self.ball_info[2], self.walkconfiguration.WalkLineMiddle_blue())
                moveTask_1 = self.motionProxy.post.moveTo(1 * self.ball_info[0], 0, 0,
                                                          self.walkconfiguration.WalkLineMiddle_blue())

    def walkToBall3(self, min_ball_d=0.4, max_ball_theta=0.15):
        isfindBall = self.findball(self.pitchAngle)
        if isfindBall is False:
            while True:
                isfindBall = self.moveheadToFindball(pitchAngles=[0, -15, 15])
                if isfindBall is False:
                    self.motionProxy.setMoveArmsEnabled(False, False)
                    self.motionProxy.moveTo(0.5, 0, 0, self.walkconfiguration.WalkLineLittle_blue())
                else:
                    break
        self.motionProxy.moveTo(0, 0, self.ball_info[2], self.walkconfiguration.WalkCircleLittle_blue())
        self.motionProxy.angleInterpolationWithSpeed("HeadYaw", 0, 0.1)

        moveTask_1 = self.motionProxy.post.moveTo(1.5 * self.ball_info[0], 0, 0,
                                                  self.walkconfiguration.WalkLineLittle_blue())
        while True:
            isfindBall = self.findball(self.pitchAngle, 0)
            if isfindBall is False:
                self.motionProxy.stop(moveTask_1)
                time.sleep(0.5)
                isfindBall_2 = self.moveheadToFindball(
                    pitchAngles=[self.pitchAngle, self.pitchAngle + 10, self.pitchAngle + 20, self.pitchAngle - 10],
                    yawAngles=[-15, 15])
                if isfindBall_2:
                    moveTask_1 = self.motionProxy.post.moveTo(1.5 * self.ball_info[0], 0, 0,
                                                              self.walkconfiguration.WalkLineLittle_blue())

            ball_d = ((self.ball_info[0] ** 2 + self.ball_info[1] ** 2) ** 0.5)
            ball_theta = abs(self.ball_info[2])
            time.sleep(0.5)

            Names = ["HeadPitch"]
            if ball_d > 0.6:
                self.pitchAngle = 0
            elif 0.4 < ball_d < 0.6:
                self.pitchAngle = 10
            elif 0.15 < ball_d < 0.4:
                self.pitchAngle = 20
            self.motionProxy.angleInterpolationWithSpeed(Names, self.pitchAngle * rad, 0.1)

            if min_ball_d / 3 < self.ball_info[0] < 0.4:
                self.tts.say("I am right")
                self.motionProxy.stop(moveTask_1)
                time.sleep(0.5)
                break

            if 0.13 < ball_d < min_ball_d:
                self.tts.say("I am right.")
                self.motionProxy.stop(moveTask_1)
                time.sleep(0.5)
                break
            elif 0.22 < abs(ball_theta) < max_ball_theta:
                if min_ball_d / 3 < ball_d < min_ball_d:
                    self.tts.say("I am right.")
                    self.motionProxy.stop(moveTask_1)
                    time.sleep(0.5)
                    break
                else:
                    continue
            elif abs(ball_theta) > max_ball_theta:
                self.tts.say("I am wrong.")
                self.motionProxy.stop(moveTask_1)
                time.sleep(0.5)
                self.motionProxy.moveTo(0, 0, self.ball_info[2], self.walkconfiguration.WalkLineMiddle_blue())
                moveTask_1 = self.motionProxy.post.moveTo(1 * self.ball_info[0], 0, 0,
                                                          self.walkconfiguration.WalkLineMiddle_blue())

    def moveCircleWithTarget(self, target_info, ball_d):
        MCAngle = 0 * rad
        # MCAngle = 0 * rad
        if target_info[0] > 0:
            target_ball_Dis = (target_info[0] ** 2 + ball_d ** 2 - 2 * target_info[0] * ball_d * math.cos(
                target_info[1])) ** 0.5
            target_ball_robotAngle = math.acos(
                (-target_info[0] ** 2 + ball_d ** 2 + target_ball_Dis ** 2) / (2 * ball_d * target_ball_Dis)) - \
                                     target_info[2]

        elif target_info[0] < 0:
            target_ball_robotAngle = math.pi - target_info[2] * rad - abs(target_info[1]) - MCAngle
            stickAngle = abs(target_info[1])
            target_ball_Dis = (ball_d * math.sin(stickAngle)) / math.sin(target_info[2] * rad)

        if target_info[1] > 0:
            self.isbackhangFlag = False
            if 0 <= target_ball_robotAngle < 0.5 * math.pi:
                moveAngle = 0.5 * math.pi - target_ball_robotAngle
                move_d = (ball_d ** 2 + ball_d ** 2 - 2 * ball_d * ball_d * math.cos(moveAngle)) ** 0.5
                self.motionProxy.moveTo(0, -move_d, moveAngle, self.walkconfiguration.WalkCircleLittle_blue())
                if self.flag is True:
                    self.motionProxy.moveTo(0, -ball_d, 0, self.walkconfiguration.WalkLineLittle_blue())
                    self.flag = False
            else:
                moveAngle = target_ball_robotAngle - 0.5 * math.pi
                move_d = (ball_d ** 2 + ball_d ** 2 - 2 * ball_d * ball_d * math.cos(moveAngle)) ** 0.5
                self.motionProxy.moveTo(0, move_d, -moveAngle, self.walkconfiguration.WalkCircleLittle_blue())
                if self.flag is True:
                    self.motionProxy.moveTo(0, ball_d, 0, self.walkconfiguration.WalkLineLittle_blue())
                    self.flag = False
            nextAngle = math.atan(target_ball_Dis / ball_d)

        elif target_info[1] < 0:
            self.isbackhangFlag = True
            if 0 <= target_ball_robotAngle < 0.5 * math.pi:
                moveAngle = 0.5 * math.pi - target_ball_robotAngle
                move_d = (ball_d ** 2 + ball_d ** 2 - 2 * ball_d * ball_d * math.cos(moveAngle)) ** 0.5
                self.motionProxy.moveTo(0, move_d, -moveAngle, self.walkconfiguration.WalkCircleLittle_blue())
                if self.flag is True:
                    self.motionProxy.moveTo(0, ball_d, 0, self.walkconfiguration.WalkLineLittle_blue())
                    self.flag = False
            else:
                moveAngle = 1.5 * math.pi - target_ball_robotAngle
                move_d = (ball_d ** 2 + ball_d ** 2 - 2 * ball_d * ball_d * math.cos(moveAngle)) ** 0.5
                self.motionProxy.moveTo(0, move_d, -moveAngle + 10 * rad,
                                        self.walkconfiguration.WalkCircleLittle_blue())
                print("move To1:", move_d)
                print("move angle1:", moveAngle * almath.TO_DEG)
                if self.flag is True:
                    self.motionProxy.moveTo(0, ball_d, 0, self.walkconfiguration.WalkLineLittle_blue())
                    self.flag = False
            nextAngle = -math.atan(target_ball_Dis / ball_d)
        print("move To2:", move_d)
        print("move angle2:", moveAngle * almath.TO_DEG)
        return nextAngle

    def makeTriangleWithTarget(self, locateTime, target_info):
        hitballFlag = False
        isfindBall = self.moveheadToFindball(pitchAngles=[20, 10, 0], yawAngles=[0])
        if isfindBall:
            pass
        else:
            self.moveheadToFindball(pitchAngles=[20, 10, 0], yawAngles=[-45, -15, 0, 15, 45])
        ball_d = (self.ball_info[0] ** 2 + self.ball_info[1] ** 2) ** 0.5
        nextAngle = 0
        if ball_d > 0:
            nextAngle = self.moveCircleWithTarget(target_info, ball_d) * rad
            for i in range(2):
                print("test1:")
                if locateTime > 8:
                    bias_x_max = 317
                    bias_y_max = 424
                    islocate = self.locateWithImage(bias_x_max, bias_y_max)
                else:
                    islocate = self.locateWithImage()
                if islocate and locateTime >= 1:
                    self.tts.say("I am ready to hit ball")
                    hitballFlag = True
                    time.sleep(0.5)
                    break
            locateTime += 1
        print("nextAngle:", nextAngle)
        return [hitballFlag, locateTime, nextAngle]

    def locateWithImage(self, best_ball_x=317, best_ball_y=424, best_ball_radius=20):
        isfindBall = self.moveheadToFindball(pitchAngles=[20, 10, 0], yawAngles=[0])
        if isfindBall is False:
            self.moveheadToFindball(pitchAngles=[0, 30, 20, 10], yawAngles=[0, -45, -15, 15, 45])
        centerX, centerY, radius = self.ballDetect.getBallInfoInImage()
        if centerX != 0 and centerY != 0:
            bias_y = centerX - best_ball_x
            bias_x = centerY - best_ball_y
            Kpx = 0.0002
            Kpy = 0.0008

            if abs(radius - best_ball_radius) < 30:
                if (abs(bias_x) < 40 and abs(bias_y) < 20):
                    print("bias_x:", bias_x)
                    print("bias_y:", bias_y)
                    self.tts.say("I am OK")
                    return True
                else:
                    move_x = Kpx * bias_x
                    move_x = -1.0 / 100 if (move_x < 0 and move_x > -1.0 / 100) else move_x
                    move_x = 1.0 / 100 if (move_x > 0 and move_x < 1.0 / 100) else move_x
                    if bias_x > 15:
                        self.motionProxy.moveTo(-move_x, 0, 0, self.walkconfiguration.WalkLineLittle_blue())
                        time.sleep(0.5)
                        print("bias_x movex1:", bias_x)
                        print("move_x1:", -move_x)
                    if bias_x < -15:
                        self.motionProxy.moveTo(-move_x, 0, 0, self.walkconfiguration.WalkLineLittle_blue())
                        time.sleep(0.5)
                        print("bias x movex2", bias_x)
                        print("move_x2", -move_x)

                    move_y = Kpy * bias_y
                    move_y = -1.0 / 100 if (move_y < 0 and move_y > -1.0 / 100) else move_y
                    move_y = 1.0 / 100 if (move_y > 0 and move_y < 1.0 / 100) else move_y
                    if abs(bias_y) > 15:
                        self.motionProxy.moveTo(0, -move_y, 0, self.walkconfiguration.WalkSideLittle_blue())
                        time.sleep(0.5)
                        print("move_y1:", -move_y)

    def compensateAngle1(self, hitballtimes):
        if hitballtimes == 1:
            compensateAngle1 = 5
        elif hitballtimes == 2:
            compensateAngle1 = 8
        elif hitballtimes == 0:
            compensateAngle1 = 3
        else:
            compensateAngle1 = 10

        return compensateAngle1

    def hitball(self, hitspeed):
        if self.landmark_info[3] > 0:
            self.forehandToHitball(hitspeed)
        elif self.landmark_info[3] < 0:
            self.backhandToHitball(hitspeed)
        elif self.stickAngle > 0:
            self.forehandToHitball(hitspeed)
        elif self.stickAngle < 0:
            self.backhandToHitball(hitspeed)

    def hitball3(self, hitspeed):
        if self.landmark_info[3] > 0:
            self.forehandToHitball(hitspeed)
            self.motionProxy.moveTo(0, 0, 80 * rad, self.walkconfiguration.WalkCircleLittle_blue())
        elif self.landmark_info[3] < 0:
            self.backhandToHitball(hitspeed)
            self.motionProxy.moveTo(0, 0, -80 * rad, self.walkconfiguration.WalkCircleLittle_blue())
        elif self.stickAngle > 0:
            self.forehandToHitball(hitspeed)
            self.motionProxy.moveTo(0, 0, 80 * rad, self.walkconfiguration.WalkCircleLittle_blue())
        elif self.stickAngle < 0:
            self.backhandToHitball(hitspeed)
            self.motionProxy.moveTo(0, 0, -80 * rad, self.walkconfiguration.WalkCircleLittle_blue())

    def trianglocation(self):
        locateTime = 0
        target_info = [0, 0, 0]
        location_info = [0, 0, 0]
        yawAnglelast = 0
        while True:
            if locateTime == 0:
                self.moveheadToFindball(pitchAngles=[20, 10, 0], yawAngles=[0, -15, -20, 15, -20])
                self.motionProxy.moveTo(0, 0, self.ball_info[2], self.walkconfiguration.WalkCircleLittle_blue())
                print("move angle:", self.ball_info[2])
                yawAnglelist = [0, -100, -80, -40, 100, 80, 40]
            else:
                yawAnglelist = [yawAnglelast + 80, yawAnglelast + 60, yawAnglelast + 20, yawAnglelast,
                                yawAnglelast - 20,
                                yawAnglelast - 60, yawAnglelast - 80]
                print("yawAnglelist start")

            target_info = self.moveheadToFindLandmarkandStick(yawAngles=yawAnglelist)
            if target_info[0] != 0:
                location_info = self.makeTriangleWithTarget(locateTime, target_info)
                if location_info[0] is True:
                    print("self.pitch:", self.pitchAngle)
                    break
                if location_info[1] > 10:
                    print("self.pitch:", self.pitchAngle)
                    self.tts.say("I am ready to hit ball")
                    break

                yawAnglelast = location_info[2]
                print("yawAngleLast: ", yawAnglelast * rad)
                locateTime = location_info[1]

    def gameTask_2WalkTask(self, moveAngle=15, distance=0.75):
        self.motionProxy.setMoveArmsEnabled(False, False)
        self.motionProxy.moveTo(0, 0, moveAngle * rad, self.walkconfiguration.WalkCircleLittle_blue())
        time.sleep(0.5)
        self.motionProxy.setMoveArmsEnabled(False, False)
        self.motionProxy.moveTo(distance, 0, 0, self.walkconfiguration.WalkLineBig_blue())
        time.sleep(0.5)
        self.motionProxy.setMoveArmsEnabled(False, False)
        self.motionProxy.moveTo(0, 0, 80 * rad, self.walkconfiguration.WalkCircleLittle_blue())
        time.sleep(0.5)

    def WalkToStick(self, distance=2, walktimes=4, min_stick_angle=10, max_stick_angle=15):
        min_stick_angle = min_stick_angle * rad
        max_stick_angle = max_stick_angle * rad
        isfind = False
        for i in range(walktimes):
            isfind = self.moveheadTofindstick2(yawAngles=[0, 40, -40])
            self.motionProxy.angleInterpolationWithSpeed("HeadYaw", 0, 0.1)
            if isfind:
                self.motionProxy.angleInterpolationWithSpeed("HeadYaw", 0, 0.1)
                if min_stick_angle < self.stickAngle < max_stick_angle:
                    self.tts.say("I am right.")
                # if min_stick_angle<self.landmark_info[2]<max_stick_angle:

                else:
                    self.motionProxy.setMoveArmsEnabled(False, False)
                    self.motionProxy.moveTo(0, 0, self.stickAngle - min_stick_angle,
                                            self.walkconfiguration.WalkCircleLittle_blue())
                    time.sleep(0.5)
            self.motionProxy.moveTo((1 / walktimes) * distance, 0, 0, self.walkconfiguration.WalkLineBig_blue())

    def gameTask_1(self):
        self.tts.say("Game Task One")

        FrontFlagTrue = 0
        MiddleFlagTrue = 0
        while True:
            FrontFlag = self.memoryProxy.getData("Device/SubDeviceList/Head/Touch/Front/Sensor/Value")
            MiddleFlag = self.memoryProxy.getData("Device/SubDeviceList/Head/Touch/Middle/Sensor/Value")
            RearFlag = self.memoryProxy.getData("Device/SubDeviceList/Head/Touch/Rear/Sensor/Value")
            if FrontFlag == 1:
                self.Ready()
                FrontFlagTrue = 1
            elif FrontFlagTrue == 1 and MiddleFlag == 1:
                self.HoldingPole()
                MiddleFlagTrue = 1
            elif FrontFlagTrue == 1 and MiddleFlagTrue == 1 and RearFlag == 1:
                self.AddressingTheBall()
                time.sleep(3)
                # self.Batting(self.firstHitSpeed_1_blue)
                self.Batting(self.firstHitSpeed_1_red)
                self.ReceivingPole()
                break
        self.motionProxy.setMoveArmsEnabled(False, False)
        self.motionProxy.moveTo(0, 0, 80 * rad, self.walkconfiguration.WalkCircleLittle_blue())
        self.hitballtimes += 3
        isfindBall = self.findball(pitchAngle=15)
        if isfindBall is False:
            self.motionProxy.moveTo(1.25, 0, 0, self.walkconfiguration.WalkLineBig_blue())
        while True:
            self.walkToBall()
            print("test")
            self.trianglocation()
            self.hitball(self.firstHitSpeed_2_blue)
            # self.hitball(self.firstHitSpeed_2_red)
            break

    def gameTask_2(self):
        self.tts.say("Game Task Two")
        self.hitballtimes += 2
        print("hitballtimes1", self.hitballtimes)
        isfindball = False
        FrontFlagTrue = 0
        MiddleFlagTrue = 0
        while True:
            FrontFlag = self.memoryProxy.getData("Device/SubDeviceList/Head/Touch/Front/Sensor/Value")
            MiddleFlag = self.memoryProxy.getData("Device/SubDeviceList/Head/Touch/Middle/Sensor/Value")
            RearFlag = self.memoryProxy.getData("Device/SubDeviceList/Head/Touch/Rear/Sensor/Value")
            if FrontFlag == 1:
                self.Ready()
                FrontFlagTrue = 1
            elif FrontFlagTrue == 1 and MiddleFlag == 1:
                self.HoldingPole()
                MiddleFlagTrue = 1
            elif FrontFlagTrue == 1 and MiddleFlagTrue == 1 and RearFlag == 1:
                self.AddressingTheBall()
                self.Batting(0.55)
                self.ReceivingPole()
                break
        self.gameTask_2WalkTask()
        self.WalkToStick(distance=2, walktimes=4, min_stick_angle=12, max_stick_angle=17)
        while True:
            print("hitballtimes2", self.hitballtimes)
            isfindball = self.moveheadToFindball(pitchAngles=[20, 10, 0, -10, -20])
            if isfindball is False:
                if self.hitballtimes == 2:
                    self.motionProxy.setMoveArmsEnabled(False, False)
                    self.motionProxy.moveTo(0.5, 0, 0, self.walkconfiguration.WalkLineLittle_blue())
                else:
                    self.motionProxy.setMoveArmsEnabled(False, False)
                    self.motionProxy.moveTo(0.25, 0, 0, self.walkconfiguration.WalkLineLittle_blue())
                    self.hitballtimes += 1
            else:
                self.walkToBall2()
                self.trianglocation()
                if self.hitballtimes == 2:
                    self.hitball(0.5)
                else:
                    self.hitball(0.6)
                self.hitballtimes += 1
                break

    def gameTask_3(self):
        self.tts.say("Game Task Three")
        isfindball = False
        FrontFlagTrue = 0
        self.hitballtimes += 1
        MiddleFlagTrue = 0
        while True:
            FrontFlag = self.memoryProxy.getData("Device/SubDeviceList/Head/Touch/Front/Sensor/Value")
            MiddleFlag = self.memoryProxy.getData("Device/SubDeviceList/Head/Touch/Middle/Sensor/Value")
            RearFlag = self.memoryProxy.getData("Device/SubDeviceList/Head/Touch/Rear/Sensor/Value")
            if FrontFlag == 1:
                self.Ready()
                FrontFlagTrue = 1
            elif FrontFlagTrue == 1 and MiddleFlag == 1:
                self.HoldingPole()
                MiddleFlagTrue = 1
            elif FrontFlagTrue == 1 and MiddleFlagTrue == 1 and RearFlag == 1:
                self.AddressingTheBall()
                self.Batting(0.73)
                self.ReceivingPole()
                break
        self.motionProxy.setMoveArmsEnabled(False, False)
        self.motionProxy.moveTo(0, 0, 120 * rad, self.walkconfiguration.WalkCircleLittle_blue())
        self.motionProxy.waitUntilMoveIsFinished()
        self.motionProxy.setMoveArmsEnabled(False, False)
        self.motionProxy.moveTo(1.6, 0, 0, self.walkconfiguration.WalkLineBig_blue())
        self.motionProxy.waitUntilMoveIsFinished()
        self.motionProxy.setMoveArmsEnabled(False, False)
        self.motionProxy.moveTo(0, 0, -120 * rad, self.walkconfiguration.WalkCircleLittle_blue())
        while True:
            if self.hitballtimes == 2:
                self.motionProxy.setMoveArmsEnabled(False, False)
                self.motionProxy.moveTo(0.70, 0, 0, self.walkconfiguration.WalkLineLittle_blue())
                self.flag = True
            self.walkToBall3()
            self.trianglocation()
            if self.hitballtimes == 1:
                self.hitball3(0.65)
            else:
                self.hitball3(0.65)
            self.hitballtimes += 1
            if self.hitballtimes == 7:
                break

    def Ready(self):
        names = list()
        times = list()
        keys = list()

        names.append("HeadPitch")
        times.append([1.36])
        keys.append([[-0.135034, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("HeadYaw")
        times.append([1.36])
        keys.append([[-0.00464392, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("LAnklePitch")
        times.append([1.36])
        keys.append([[0.0886833, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("LAnkleRoll")
        times.append([1.36])
        keys.append([[-0.108787, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("LElbowRoll")
        times.append([1.36])
        keys.append([[-0.80991, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("LElbowYaw")
        times.append([1.36])
        keys.append([[-1.51563, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("LHand")
        times.append([1.36])
        keys.append([[0.8848, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("LHipPitch")
        times.append([1.36])
        keys.append([[0.126656, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("LHipRoll")
        times.append([1.36])
        keys.append([[0.116344, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("LHipYawPitch")
        times.append([1.36])
        keys.append([[-0.17046, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("LKneePitch")
        times.append([1.36])
        keys.append([[-0.0923279, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("LShoulderPitch")
        times.append([1.36])
        keys.append([[0.771559, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("LShoulderRoll")
        times.append([1.36])
        keys.append([[0.0797259, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("LWristYaw")
        times.append([1.36])
        keys.append([[-0.188724, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("RAnklePitch")
        times.append([1.36])
        keys.append([[0.0874194, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("RAnkleRoll")
        times.append([1.36])
        keys.append([[0.109102, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("RElbowRoll")
        times.append([1.36])
        keys.append([[0.621311, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("RElbowYaw")
        times.append([1.36])
        keys.append([[1.52322, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("RHand")
        times.append([1.36])
        keys.append([[0.998363, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("RHipPitch")
        times.append([1.36])
        keys.append([[0.126321, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("RHipRoll")
        times.append([1.36])
        keys.append([[-0.11736, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("RHipYawPitch")
        times.append([1.36])
        keys.append([[0.737377, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("RKneePitch")
        times.append([1.36])
        keys.append([[-0.0894577, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("RShoulderPitch")
        times.append([1.36])
        keys.append([[0.65506, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("RShoulderRoll")
        times.append([1.36])
        keys.append([[0.0106959, [3, -0.466667, 0], [3, 0, 0]]])

        names.append("RWristYaw")
        times.append([1.36])
        keys.append([[0.00916195, [3, -0.466667, 0], [3, 0, 0]]])

        try:
            motion = ALProxy("ALMotion", self.IP, self.PORT)
            motion.angleInterpolationBezier(names, times, keys)
        except BaseException, err:
            print
            err

    def HoldingPole(self):
        names = list()
        times = list()
        keys = list()

        names.append("HeadPitch")
        times.append([0.4])
        keys.append([[-0.135034, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("HeadYaw")
        times.append([0.4])
        keys.append([[-0.00464392, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("LAnklePitch")
        times.append([0.4])
        keys.append([[0.0886833, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("LAnkleRoll")
        times.append([0.4])
        keys.append([[-0.108787, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("LElbowRoll")
        times.append([0.4, 0.96, 1.4, 2, 2.4, 2.96])
        keys.append([[-0.80991, [3, -0.146667, 0], [3, 0.186667, 0]],
                     [-0.811062, [3, -0.186667, 0.00115195], [3, 0.146667, -0.000905104]],
                     [-0.819703, [3, -0.146667, 0.00252602], [3, 0.2, -0.00344457]],
                     [-0.828973, [3, -0.2, 0], [3, 0.133333, 0]], [-0.819703, [3, -0.133333, 0], [3, 0.186667, 0]],
                     [-1.53923, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("LElbowYaw")
        times.append([0.4, 0.96, 1.4, 2, 2.4, 2.96])
        keys.append([[-1.51563, [3, -0.146667, 0], [3, 0.186667, 0]],
                     [-1.52345, [3, -0.186667, 0.00331594], [3, 0.146667, -0.00260538]],
                     [-1.5334, [3, -0.146667, 0.00137373], [3, 0.2, -0.00187327]],
                     [-1.53527, [3, -0.2, 0], [3, 0.133333, 0]], [-1.5334, [3, -0.133333, 0], [3, 0.186667, 0]],
                     [-1.73363, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("LHand")
        times.append([0.4, 0.96, 1.4, 2, 2.4, 2.96])
        keys.append([[0.8848, [3, -0.146667, 0], [3, 0.186667, 0]],
                     [0.884792, [3, -0.186667, 8.04663e-06], [3, 0.146667, -6.32235e-06]],
                     [0.00413917, [3, -0.146667, 0], [3, 0.2, 0]],
                     [0.00539452, [3, -0.2, -0.000549448], [3, 0.133333, 0.000366299]],
                     [0.00688641, [3, -0.133333, -0.00149189], [3, 0.186667, 0.00208864]],
                     [0.526548, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("LHipPitch")
        times.append([0.4])
        keys.append([[0.126656, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("LHipRoll")
        times.append([0.4])
        keys.append([[0.116344, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("LHipYawPitch")
        times.append([0.4])
        keys.append([[-0.17046, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("LKneePitch")
        times.append([0.4])
        keys.append([[-0.0923279, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("LShoulderPitch")
        times.append([0.4, 0.96, 1.4, 2, 2.4, 2.96])
        keys.append([[0.771559, [3, -0.146667, 0], [3, 0.186667, 0]],
                     [0.774808, [3, -0.186667, -0.00206391], [3, 0.146667, 0.00162164]],
                     [0.782616, [3, -0.146667, -0.00216065], [3, 0.2, 0.00294634]],
                     [0.790128, [3, -0.2, 0], [3, 0.133333, 0]], [0.782616, [3, -0.133333, 0], [3, 0.186667, 0]],
                     [2.01497, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("LShoulderRoll")
        times.append([0.4, 0.96, 1.4, 2, 2.4, 2.96])
        keys.append([[0.0797259, [3, -0.146667, 0], [3, 0.186667, 0]],
                     [0.0788562, [3, -0.186667, 0.000647972], [3, 0.146667, -0.000509121]],
                     [0.0762547, [3, -0.146667, 0.00133235], [3, 0.2, -0.00181685]],
                     [0.0694086, [3, -0.2, 0], [3, 0.133333, 0]], [0.0762547, [3, -0.133333, 0], [3, 0.186667, 0]],
                     [0.0264343, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("LWristYaw")
        times.append([0.4, 0.96, 1.4, 2, 2.4, 2.96])
        keys.append([[-0.188724, [3, -0.146667, 0], [3, 0.186667, 0]], [-0.189255, [3, -0.186667, 0], [3, 0.146667, 0]],
                     [-0.0167174, [3, -0.146667, 0], [3, 0.2, 0]], [-1.82049, [3, -0.2, 0], [3, 0.133333, 0]],
                     [-1.54395, [3, -0.133333, -0.0944333], [3, 0.186667, 0.132207]],
                     [-1.14057, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("RAnklePitch")
        times.append([0.4])
        keys.append([[0.0874194, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("RAnkleRoll")
        times.append([0.4])
        keys.append([[0.109102, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("RElbowRoll")
        times.append([0.4, 0.96, 1.4, 2, 2.4, 2.96])
        keys.append([[0.621311, [3, -0.146667, 0], [3, 0.186667, 0]],
                     [0.765735, [3, -0.186667, -0.0360534], [3, 0.146667, 0.0283277]],
                     [0.814454, [3, -0.146667, -0.00510808], [3, 0.2, 0.00696557]],
                     [0.82142, [3, -0.2, 0], [3, 0.133333, 0]], [0.812292, [3, -0.133333, 0], [3, 0.186667, 0]],
                     [1.5392, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("RElbowYaw")
        times.append([0.4, 0.96, 1.4, 2, 2.4, 2.96])
        keys.append([[1.52322, [3, -0.146667, 0], [3, 0.186667, 0]], [1.53252, [3, -0.186667, 0], [3, 0.146667, 0]],
                     [1.53203, [3, -0.146667, 0.000492152], [3, 0.2, -0.000671117]],
                     [1.52334, [3, -0.2, 0], [3, 0.133333, 0]],
                     [1.53203, [3, -0.133333, -0.00582284], [3, 0.186667, 0.00815197]],
                     [1.56527, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("RHand")
        times.append([0.4, 0.96, 1.4, 2, 2.4, 2.96])
        keys.append([[0.998363, [3, -0.146667, 0], [3, 0.186667, 0]], [0.998363, [3, -0.186667, 0], [3, 0.146667, 0]],
                     [0.995963, [3, -0.146667, 0.000480046], [3, 0.2, -0.000654608]],
                     [0.994959, [3, -0.2, 0.00100392], [3, 0.133333, -0.000669281]],
                     [0, [3, -0.133333, 0], [3, 0.186667, 0]], [0.0060968, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("RHipPitch")
        times.append([0.4])
        keys.append([[0.126321, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("RHipRoll")
        times.append([0.4])
        keys.append([[-0.11736, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("RHipYawPitch")
        times.append([0.4])
        keys.append([[0.737377, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("RKneePitch")
        times.append([0.4])
        keys.append([[-0.0894577, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("RShoulderPitch")
        times.append([0.4, 0.96, 1.4, 2, 2.4, 2.96])
        keys.append([[0.65506, [3, -0.146667, 0], [3, 0.186667, 0]], [0.822434, [3, -0.186667, 0], [3, 0.146667, 0]],
                     [0.77649, [3, -0.146667, 0.00131187], [3, 0.2, -0.00178891]],
                     [0.774701, [3, -0.2, 0], [3, 0.133333, 0]],
                     [0.779286, [3, -0.133333, -0.0045849], [3, 0.186667, 0.00641886]],
                     [1.98158, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("RShoulderRoll")
        times.append([0.4, 0.96, 1.4, 2, 2.4, 2.96])
        keys.append(
            [[0.0106959, [3, -0.146667, 0], [3, 0.186667, 0]], [0.00781977, [3, -0.186667, 0], [3, 0.146667, 0]],
             [0.0682579, [3, -0.146667, 0], [3, 0.2, 0]], [0.0505979, [3, -0.2, 0], [3, 0.133333, 0]],
             [0.0554728, [3, -0.133333, 0], [3, 0.186667, 0]], [-0.0834505, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("RWristYaw")
        times.append([0.4, 0.96, 1.4, 2, 2.4, 2.96])
        keys.append([[0.00916195, [3, -0.146667, 0], [3, 0.186667, 0]],
                     [0.00862908, [3, -0.186667, 0.000532867], [3, 0.146667, -0.000418681]],
                     [-1.58205, [3, -0.146667, 0.000378977], [3, 0.2, -0.000516786]],
                     [-1.58257, [3, -0.2, 0], [3, 0.133333, 0]], [-1.58112, [3, -0.133333, 0], [3, 0.186667, 0]],
                     [-1.66805, [3, -0.186667, 0], [3, 0, 0]]])

        try:
            motion = ALProxy("ALMotion", self.IP, self.PORT)
            motion.angleInterpolationBezier(names, times, keys)
        except BaseException, err:
            print
            err

    def AddressingTheBall(self):
        names = list()
        times = list()
        keys = list()

        names.append("LElbowRoll")
        times.append([0.56, 1.08, 1.52])
        keys.append([[-1.54341, [3, -0.2, 0], [3, 0.173333, 0]], [-0.822267, [3, -0.173333, 0], [3, 0.146667, 0]],
                     [-0.83222, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("LElbowYaw")
        times.append([0.56, 1.08, 1.52])
        keys.append([[-1.71474, [3, -0.2, 0], [3, 0.173333, 0]], [-1.53416, [3, -0.173333, 0], [3, 0.146667, 0]],
                     [-1.54126, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("LHand")
        times.append([0.56, 1.08, 1.52])
        keys.append([[0.5268, [3, -0.2, 0], [3, 0.173333, 0]], [0.00439295, [3, -0.173333, 0], [3, 0.146667, 0]],
                     [0.00459188, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("LShoulderPitch")
        times.append([0.56, 1.08, 1.52])
        keys.append([[2.01823, [3, -0.2, 0], [3, 0.173333, 0]], [0.784998, [3, -0.173333, 0], [3, 0.146667, 0]],
                     [0.788002, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("LShoulderRoll")
        times.append([0.56, 1.08, 1.52])
        keys.append([[-0.0402248, [3, -0.2, 0], [3, 0.173333, 0]], [0.0738154, [3, -0.173333, 0], [3, 0.146667, 0]],
                     [0.0683689, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("LWristYaw")
        times.append([0.56, 1.08, 1.52])
        keys.append([[-1.13827, [3, -0.2, 0], [3, 0.173333, 0]], [-1.70836, [3, -0.173333, 0], [3, 0.146667, 0]],
                     [-1.70195, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("RElbowRoll")
        times.append([0.56, 1.08, 1.52])
        keys.append([[1.54339, [3, -0.2, 0], [3, 0.173333, 0]], [0.816889, [3, -0.173333, 0], [3, 0.146667, 0]],
                     [0.826612, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("RElbowYaw")
        times.append([0.56, 1.08, 1.52])
        keys.append(
            [[1.54622, [3, -0.2, 0], [3, 0.173333, 0]], [1.53, [3, -0.173333, 0.00434004], [3, 0.146667, -0.00367234]],
             [1.52218, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("RHand")
        times.append([0.56, 1.08, 1.52])
        keys.append([[0, [3, -0.2, 0], [3, 0.173333, 0]], [0, [3, -0.173333, 0], [3, 0.146667, 0]],
                     [0.99973, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("RShoulderPitch")
        times.append([0.56, 1.08, 1.52])
        keys.append([[2.00566, [3, -0.2, 0], [3, 0.173333, 0]], [0.778104, [3, -0.173333, 0], [3, 0.146667, 0]],
                     [0.780231, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("RShoulderRoll")
        times.append([0.56, 1.08, 1.52])
        keys.append([[-0.025278, [3, -0.2, 0], [3, 0.173333, 0]], [0.0570375, [3, -0.173333, 0], [3, 0.146667, 0]],
                     [0.0470152, [3, -0.146667, 0], [3, 0, 0]]])

        names.append("RWristYaw")
        times.append([0.56, 1.08, 1.52])
        keys.append([[-1.66903, [3, -0.2, 0], [3, 0.173333, 0]], [-1.58255, [3, -0.173333, 0], [3, 0.146667, 0]],
                     [-1.59122, [3, -0.146667, 0], [3, 0, 0]]])

        try:
            motion = ALProxy("ALMotion", self.IP, self.PORT)
            motion.angleInterpolationBezier(names, times, keys)
        except BaseException, err:
            print
            err

    def Batting(self, speed):
        names = list()
        times = list()
        keys = list()

        names.append("LElbowRoll")
        times.append([0, speed])
        keys.append([[-0.822266, [3, -0.0133333, 0], [3, 0.186667, 0]], [-0.819703, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("LElbowYaw")
        times.append([0, speed])
        keys.append([[-1.53416, [3, -0.0133333, 0], [3, 0.186667, 0]], [-1.5334, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("LHand")
        times.append([0, speed])
        keys.append([[0.00439295, [3, -0.0133333, 0], [3, 0.186667, 0]], [0.00413917, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("LShoulderPitch")
        times.append([0, speed])
        keys.append([[0.784999, [3, -0.0133333, 0], [3, 0.186667, 0]], [0.782616, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("LShoulderRoll")
        times.append([0, speed])
        keys.append([[0.0738153, [3, -0.0133333, 0], [3, 0.186667, 0]], [0.0762547, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("LWristYaw")
        times.append([0, speed])
        keys.append([[-1.70836, [3, -0.0133333, 0], [3, 0.186667, 0]], [-0.0167174, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("RElbowRoll")
        times.append([0, speed])
        keys.append([[0.816889, [3, -0.0133333, 0], [3, 0.186667, 0]], [0.814454, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("RElbowYaw")
        times.append([0, speed])
        keys.append([[1.53, [3, -0.0133333, 0], [3, 0.186667, 0]], [1.53203, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("RHand")
        times.append([0, speed])
        keys.append([[0.995607, [3, -0.0133333, 0], [3, 0.186667, 0]], [0.995963, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("RShoulderPitch")
        times.append([0, speed])
        keys.append([[0.778105, [3, -0.0133333, 0], [3, 0.186667, 0]], [0.77649, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("RShoulderRoll")
        times.append([0, speed])
        keys.append([[0.0570375, [3, -0.0133333, 0], [3, 0.186667, 0]], [0.0682579, [3, -0.186667, 0], [3, 0, 0]]])

        names.append("RWristYaw")
        times.append([0, speed])
        keys.append([[-1.58255, [3, -0.0133333, 0], [3, 0.186667, 0]], [-1.58205, [3, -0.186667, 0], [3, 0, 0]]])

        try:
            motion = ALProxy("ALMotion", self.IP, self.PORT)
            motion.angleInterpolationBezier(names, times, keys)
        except BaseException, err:
            print
            err

    def ReceivingPole(self):
        names = list()
        times = list()
        keys = list()

        names.append("HeadPitch")
        times.append([0.44, 1.28])
        keys.append([[-0.13086, [3, -0.16, 0], [3, 0.28, 0]], [-0.13086, [3, -0.28, 0], [3, 0, 0]]])

        names.append("HeadYaw")
        times.append([0.44, 1.28])
        keys.append([[-0.00781646, [3, -0.16, 0], [3, 0.28, 0]], [-0.00781646, [3, -0.28, 0], [3, 0, 0]]])

        names.append("LElbowRoll")
        times.append([0.44, 1.28, 1.96])
        keys.append(
            [[-0.819703, [3, -0.16, 0], [3, 0.28, 0]], [-0.823499, [3, -0.28, 0.00379607], [3, 0.226667, -0.00307301]],
             [-1.53812, [3, -0.226667, 0], [3, 0, 0]]])

        names.append("LElbowYaw")
        times.append([0.44, 1.28, 1.96])
        keys.append(
            [[-1.5334, [3, -0.16, 0], [3, 0.28, 0]], [-1.53908, [3, -0.28, 0.00568625], [3, 0.226667, -0.00460315]],
             [-1.72773, [3, -0.226667, 0], [3, 0, 0]]])

        names.append("LHand")
        times.append([0.44, 1.28, 1.96])
        keys.append([[0.00688641, [3, -0.16, 0], [3, 0.28, 0]],
                     [0.00689399, [3, -0.28, -7.58003e-06], [3, 0.226667, 6.13622e-06]],
                     [0.526673, [3, -0.226667, 0], [3, 0, 0]]])

        names.append("LShoulderPitch")
        times.append([0.44, 1.28, 1.96])
        keys.append(
            [[0.782616, [3, -0.16, 0], [3, 0.28, 0]], [0.790191, [3, -0.28, -0.00757476], [3, 0.226667, 0.00613195]],
             [2.00793, [3, -0.226667, 0], [3, 0, 0]]])

        names.append("LShoulderRoll")
        times.append([0.44, 1.28, 1.96])
        keys.append(
            [[0.0762547, [3, -0.16, 0], [3, 0.28, 0]], [0.0687017, [3, -0.28, 0.00755291], [3, 0.226667, -0.00611426]],
             [0.0290275, [3, -0.226667, 0], [3, 0, 0]]])

        names.append("LWristYaw")
        times.append([0.44, 1.28, 1.96])
        keys.append([[-1.7683, [3, -0.16, 0], [3, 0.28, 0]], [-1.78024, [3, -0.28, 0], [3, 0.226667, 0]],
                     [-1.14443, [3, -0.226667, 0], [3, 0, 0]]])

        names.append("RElbowRoll")
        times.append([0.44, 1.28, 1.96])
        keys.append(
            [[0.812292, [3, -0.16, 0], [3, 0.28, 0]], [0.814847, [3, -0.28, -0.00255517], [3, 0.226667, 0.00206847]],
             [1.53801, [3, -0.226667, 0], [3, 0, 0]]])

        names.append("RElbowYaw")
        times.append([0.44, 1.28, 1.96])
        keys.append(
            [[1.53203, [3, -0.16, 0], [3, 0.28, 0]], [1.53849, [3, -0.28, -0.0064595], [3, 0.226667, 0.00522912]],
             [1.5705, [3, -0.226667, 0], [3, 0, 0]]])

        names.append("RHand")
        times.append([0.44, 1.28, 1.96])
        keys.append([[0.995963, [3, -0.16, 0], [3, 0.28, 0]], [0, [3, -0.28, 0], [3, 0.226667, 0]],
                     [0.000917273, [3, -0.226667, 0], [3, 0, 0]]])

        names.append("RShoulderPitch")
        times.append([0.44, 1.28, 1.96])
        keys.append(
            [[0.779286, [3, -0.16, 0], [3, 0.28, 0]], [0.781223, [3, -0.28, -0.00193732], [3, 0.226667, 0.0015683]],
             [1.9869, [3, -0.226667, 0], [3, 0, 0]]])

        names.append("RShoulderRoll")
        times.append([0.44, 1.28, 1.96])
        keys.append(
            [[0.0554728, [3, -0.16, 0], [3, 0.28, 0]], [0.0478072, [3, -0.28, 0.00766566], [3, 0.226667, -0.00620553]],
             [-0.0770126, [3, -0.226667, 0], [3, 0, 0]]])

        names.append("RWristYaw")
        times.append([0.44, 1.28, 1.96])
        keys.append(
            [[-1.58112, [3, -0.16, 0], [3, 0.28, 0]], [-1.59005, [3, -0.28, 0.00892918], [3, 0.226667, -0.00722838]],
             [-1.66302, [3, -0.226667, 0], [3, 0, 0]]])

        try:
            motion = ALProxy("ALMotion", self.IP, self.PORT)
            motion.angleInterpolationBezier(names, times, keys)
        except BaseException, err:
            print
            err

    def AddressingTheBall2(self):
        names = list()
        times = list()
        keys = list()

        names.append("LElbowRoll")
        times.append([0.48, 0.92, 1.4, 1.92, 2.36, 2.88, 3.44, 4.04])
        keys.append([[-1.54341, [3, -0.173333, 0], [3, 0.146667, 0]], [-0.822267, [3, -0.146667, 0], [3, 0.16, 0]],
                     [-0.83222, [3, -0.16, 0.00198297], [3, 0.173333, -0.00214822]],
                     [-0.83466, [3, -0.173333, 0], [3, 0.146667, 0]], [-0.83466, [3, -0.146667, 0], [3, 0.173333, 0]],
                     [-0.83466, [3, -0.173333, 0], [3, 0.186667, 0]], [-0.83466, [3, -0.186667, 0], [3, 0.2, 0]],
                     [-0.844776, [3, -0.2, 0], [3, 0, 0]]])

        names.append("LElbowYaw")
        times.append([0.48, 0.92, 1.4, 1.92, 2.36, 2.88, 3.44, 4.04])
        keys.append([[-1.71474, [3, -0.173333, 0], [3, 0.146667, 0]], [-1.53416, [3, -0.146667, 0], [3, 0.16, 0]],
                     [-1.54126, [3, -0.16, 0], [3, 0.173333, 0]], [-1.53233, [3, -0.173333, 0], [3, 0.146667, 0]],
                     [-1.53233, [3, -0.146667, 0], [3, 0.173333, 0]], [-1.53233, [3, -0.173333, 0], [3, 0.186667, 0]],
                     [-1.53233, [3, -0.186667, 0], [3, 0.2, 0]], [-1.48139, [3, -0.2, 0], [3, 0, 0]]])

        names.append("LHand")
        times.append([0.48, 0.92, 1.4, 1.92, 2.36, 2.88, 3.44, 4.04])
        keys.append([[0.5268, [3, -0.173333, 0], [3, 0.146667, 0]], [0.00439295, [3, -0.146667, 0], [3, 0.16, 0]],
                     [0.00459188, [3, -0.16, -4.04784e-05], [3, 0.173333, 4.38516e-05]],
                     [0.00464594, [3, -0.173333, 0], [3, 0.146667, 0]],
                     [0.00464594, [3, -0.146667, 0], [3, 0.173333, 0]],
                     [0.00464594, [3, -0.173333, 0], [3, 0.186667, 0]], [0.00464594, [3, -0.186667, 0], [3, 0.2, 0]],
                     [0.00581758, [3, -0.2, 0], [3, 0, 0]]])

        names.append("LShoulderPitch")
        times.append([0.48, 0.92, 1.4, 1.92, 2.36, 2.88, 3.44, 4.04])
        keys.append([[2.01823, [3, -0.173333, 0], [3, 0.146667, 0]], [0.784998, [3, -0.146667, 0], [3, 0.16, 0]],
                     [0.788002, [3, -0.16, -0.00144039], [3, 0.173333, 0.00156042]],
                     [0.794001, [3, -0.173333, 0], [3, 0.146667, 0]], [0.794001, [3, -0.146667, 0], [3, 0.173333, 0]],
                     [0.093491, [3, -0.173333, 0], [3, 0.186667, 0]], [0.093491, [3, -0.186667, 0], [3, 0.2, 0]],
                     [0.873171, [3, -0.2, 0], [3, 0, 0]]])

        names.append("LShoulderRoll")
        times.append([0.48, 0.92, 1.4, 1.92, 2.36, 2.88, 3.44, 4.04])
        keys.append([[-0.0402248, [3, -0.173333, 0], [3, 0.146667, 0]], [0.0738154, [3, -0.146667, 0], [3, 0.16, 0]],
                     [0.0683689, [3, -0.16, 0], [3, 0.173333, 0]], [0.0782558, [3, -0.173333, 0], [3, 0.146667, 0]],
                     [0.0782558, [3, -0.146667, 0], [3, 0.173333, 0]], [0.0782558, [3, -0.173333, 0], [3, 0.186667, 0]],
                     [0.0782558, [3, -0.186667, 0], [3, 0.2, 0]], [-0.0266295, [3, -0.2, 0], [3, 0, 0]]])

        names.append("LWristYaw")
        times.append([0.48, 0.92, 1.4, 1.92, 2.36, 2.88, 3.44, 4.04])
        keys.append([[-1.13827, [3, -0.173333, 0], [3, 0.146667, 0]], [-1.70836, [3, -0.146667, 0], [3, 0.16, 0]],
                     [-1.70195, [3, -0.16, -0.00230355], [3, 0.173333, 0.00249551]],
                     [-1.69396, [3, -0.173333, 0], [3, 0.146667, 0]], [-1.69396, [3, -0.146667, 0], [3, 0.173333, 0]],
                     [-1.69396, [3, -0.173333, 0], [3, 0.186667, 0]],
                     [-0.0700772, [3, -0.186667, -0.0441683], [3, 0.2, 0.0473232]],
                     [-0.022754, [3, -0.2, 0], [3, 0, 0]]])

        names.append("RElbowRoll")
        times.append([0.48, 0.92, 1.4, 1.92, 2.36, 2.88, 3.44, 4.04])
        keys.append([[1.54339, [3, -0.173333, 0], [3, 0.146667, 0]], [0.816889, [3, -0.146667, 0], [3, 0.16, 0]],
                     [0.826612, [3, -0.16, -0.00972327], [3, 0.173333, 0.0105335]],
                     [1.35599, [3, -0.173333, 0], [3, 0.146667, 0]], [1.35599, [3, -0.146667, 0], [3, 0.173333, 0]],
                     [1.35599, [3, -0.173333, 0], [3, 0.186667, 0]], [1.35599, [3, -0.186667, 0], [3, 0.2, 0]],
                     [0.819504, [3, -0.2, 0], [3, 0, 0]]])

        names.append("RElbowYaw")
        times.append([0.48, 0.92, 1.4, 1.92, 2.36, 2.88, 3.44, 4.04])
        keys.append(
            [[1.54622, [3, -0.173333, 0], [3, 0.146667, 0]], [1.53, [3, -0.146667, 0.00383195], [3, 0.16, -0.00418031]],
             [1.52218, [3, -0.16, 0], [3, 0.173333, 0]], [1.5245, [3, -0.173333, 0], [3, 0.146667, 0]],
             [1.5245, [3, -0.146667, 0], [3, 0.173333, 0]], [1.5245, [3, -0.173333, 0], [3, 0.186667, 0]],
             [1.5245, [3, -0.186667, 0], [3, 0.2, 0]], [1.52307, [3, -0.2, 0], [3, 0, 0]]])

        names.append("RHand")
        times.append([0.48, 0.92, 1.4, 1.92, 2.36, 2.88, 3.44, 4.04])
        keys.append([[0, [3, -0.173333, 0], [3, 0.146667, 0]], [0, [3, -0.146667, 0], [3, 0.16, 0]],
                     [0.99973, [3, -0.16, 0], [3, 0.173333, 0]], [0.991605, [3, -0.173333, 0], [3, 0.146667, 0]],
                     [0.991605, [3, -0.146667, 0], [3, 0.173333, 0]], [0.991605, [3, -0.173333, 0], [3, 0.186667, 0]],
                     [0.991605, [3, -0.186667, 0], [3, 0.2, 0]], [0.992216, [3, -0.2, 0], [3, 0, 0]]])

        names.append("RShoulderPitch")
        times.append([0.48, 0.92, 1.4, 1.92, 2.36, 2.88, 3.44, 4.04])
        keys.append([[2.00566, [3, -0.173333, 0], [3, 0.146667, 0]], [0.778104, [3, -0.146667, 0], [3, 0.16, 0]],
                     [0.780232, [3, -0.16, 0], [3, 0.173333, 0]], [0.153451, [3, -0.173333, 0], [3, 0.146667, 0]],
                     [0.153451, [3, -0.146667, 0], [3, 0.173333, 0]], [0.153451, [3, -0.173333, 0], [3, 0.186667, 0]],
                     [0.153451, [3, -0.186667, 0], [3, 0.2, 0]], [0.770015, [3, -0.2, 0], [3, 0, 0]]])

        names.append("RShoulderRoll")
        times.append([0.48, 0.92, 1.4, 1.92, 2.36, 2.88, 3.44, 4.04])
        keys.append([[-0.025278, [3, -0.173333, 0], [3, 0.146667, 0]], [0.0570375, [3, -0.146667, 0], [3, 0.16, 0]],
                     [0.0470152, [3, -0.16, 0], [3, 0.173333, 0]], [0.0486284, [3, -0.173333, 0], [3, 0.146667, 0]],
                     [0.0486284, [3, -0.146667, 0], [3, 0.173333, 0]], [0.0486284, [3, -0.173333, 0], [3, 0.186667, 0]],
                     [0.0486284, [3, -0.186667, 0], [3, 0.2, 0]], [0.048196, [3, -0.2, 0], [3, 0, 0]]])

        names.append("RWristYaw")
        times.append([0.48, 0.92, 1.4, 1.92, 2.36, 2.88, 3.44, 4.04])
        keys.append([[-1.66903, [3, -0.173333, 0], [3, 0.146667, 0]], [-1.58255, [3, -0.146667, 0], [3, 0.16, 0]],
                     [-1.59122, [3, -0.16, 0], [3, 0.173333, 0]], [-1.58928, [3, -0.173333, 0], [3, 0.146667, 0]],
                     [-1.58928, [3, -0.146667, 0], [3, 0.173333, 0]], [-1.58928, [3, -0.173333, 0], [3, 0.186667, 0]],
                     [-1.58928, [3, -0.186667, 0], [3, 0.2, 0]], [-1.59736, [3, -0.2, 0], [3, 0, 0]]])

        try:
            motion = ALProxy("ALMotion", self.IP, self.PORT)
            motion.angleInterpolationBezier(names, times, keys)
        except BaseException, err:
            print
            err

    def Batting2(self, speed):
        names = list()
        times = list()
        keys = list()

        names.append("LElbowRoll")
        times.append([0, speed])
        keys.append([[-0.844776, [3, -0.0133333, 0], [3, 0.12, 0]], [-0.82291, [3, -0.12, 0], [3, 0, 0]]])

        names.append("LElbowYaw")
        times.append([0, speed])
        keys.append([[-1.48139, [3, -0.0133333, 0], [3, 0.12, 0]], [-1.53233, [3, -0.12, 0], [3, 0, 0]]])

        names.append("LHand")
        times.append([0, speed])
        keys.append([[0.00581758, [3, -0.0133333, 0], [3, 0.12, 0]], [0.00464594, [3, -0.12, 0], [3, 0, 0]]])

        names.append("LShoulderPitch")
        times.append([0, speed])
        keys.append([[0.87317, [3, -0.0133333, 0], [3, 0.12, 0]], [0.776478, [3, -0.12, 0], [3, 0, 0]]])

        names.append("LShoulderRoll")
        times.append([0, speed])
        keys.append([[-0.0266295, [3, -0.0133333, 0], [3, 0.12, 0]], [-0.0331613, [3, -0.12, 0], [3, 0, 0]]])

        names.append("LWristYaw")
        times.append([0, speed])
        keys.append([[-0.022754, [3, -0.0133333, 0], [3, 0.12, 0]], [-1.78322, [3, -0.12, 0], [3, 0, 0]]])

        names.append("RElbowRoll")
        times.append([0, speed])
        keys.append([[0.819503, [3, -0.0133333, 0], [3, 0.12, 0]], [0.818825, [3, -0.12, 0], [3, 0, 0]]])

        names.append("RElbowYaw")
        times.append([0, speed])
        keys.append([[1.52307, [3, -0.0133333, 0], [3, 0.12, 0]], [1.5245, [3, -0.12, 0], [3, 0, 0]]])

        names.append("RHand")
        times.append([0, speed])
        keys.append([[0.992216, [3, -0.0133333, 0], [3, 0.12, 0]], [0.991605, [3, -0.12, 0], [3, 0, 0]]])

        names.append("RShoulderPitch")
        times.append([0, speed])
        keys.append([[0.770016, [3, -0.0133333, 0], [3, 0.12, 0]], [0.766861, [3, -0.12, 0], [3, 0, 0]]])

        names.append("RShoulderRoll")
        times.append([0, speed])
        keys.append([[0.048196, [3, -0.0133333, 0], [3, 0.12, 0]], [0.0597576, [3, -0.12, 0], [3, 0, 0]]])

        names.append("RWristYaw")
        times.append([0, speed])
        keys.append([[-1.59736, [3, -0.0133333, 0], [3, 0.12, 0]], [-1.58928, [3, -0.12, 0], [3, 0, 0]]])

        try:
            motion = ALProxy("ALMotion", self.IP, self.PORT)
            motion.angleInterpolationBezier(names, times, keys)
        except BaseException, err:
            print
            err


def run():
    memory = ALProxy("ALMemory", "192.168.137.117", 9559)
    tts = ALProxy("ALTextToSpeech", "192.168.137.117", 9559)

    shutdown = 0
    resetFlag = 0
    while True:
        shutdown += memory.getData("Device/SubDeviceList/Head/Touch/Front/Sensor/Value")
        resetFlag += memory.getData("Device/SubDeviceList/Head/Touch/Rear/Sensor/Value")
        if shutdown > 2:
            tts.say("Over")
            os._exit(0)
        if resetFlag > 2:
            tts.say("Again,game fighting")
            python = sys.executable
            os.execl(python, python, *sys.argv)

        time.sleep(1)


if __name__ == "__main__":
    thr = threading.Thread(target=run)
    thr.start()
    action = GolfGame("192.168.137.117", 9559)
    action.gameStart()
    # action.gameTask_2WalkTask()
    # motion = ALProxy("ALMotion", "192.168.43.39", 9559)
    # motion.moveTo(0.5, 0,0, action.walkconfiguration.WalkLineLittle_blue())
    # time.sleep(10)
    # action.ReceivingPole()
    # while True:
    # time.sleep(2)
    # action.gamestart1()
    # action.forehandToHitball(0.3)
    # action.walkingWithGolfpole()
