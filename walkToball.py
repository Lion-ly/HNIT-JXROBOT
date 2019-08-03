# coding=utf-8
import math
from VisualBasis import *
from naoqi import ALProxy
from PIL import Image
import math
import almath
import time
import sys
import numpy as np
from multiprocessing import Process, Pool, Queue

robotIP = "192.168.1.105"

_stepStatue = [["MaxStepX", 0.04], ["MaxStepY", 0.14], ["MaxStepTheta", 0.4], ["MaxStepFrequency", 0.6],
               ["StepHeight", 0.02], ["TorsoWx", 0], ["TorsoWy", 0]]


class WalkToBall(object):
    def __init__(self, IP=robotIP, PORT=9559):
        self.ballDetect = BallDetect(IP=robotIP, resolution=vd.kVGA)
        self.tts = ALProxy("ALTextToSpeech", IP, PORT)
        self.camera = ALProxy("ALVideoDevice", IP, PORT)
        self.motion = ALProxy("ALMotion", IP, PORT)
        self.posture = ALProxy("ALRobotPosture", IP, PORT)

    def start(self):
        self.motion.wakeUp()
        self.motion.setStiffnesses("Head", 1.0)
        time.sleep(1.0)
        rnumber = 1
        self.camera.setActiveCamera(1)
        self.motion.setMoveArmsEnabled(False, False)
        # self.tts.say("红球在哪里呢？")
        isenabled = True
        Headnames = "HeadYaw"
        while 1:
            angle = 0
            Headnames = "HeadYaw"
            self.motion.setAngles(Headnames, angle, 0.2)
            time.sleep(1.5)
            ballData = self.ballDetect.updateBallData(colorSpace="HSV", standState="standUp", saveFrameBin=True)
            if ballData != {}:
                # self.tts.say("我看到红球了")
                break
            else:
                for i in range(1, 3):
                    if i <= 1:
                        angle -= 60 * almath.TO_RAD
                    else:
                        angle = 0
                        angle += 60 * almath.TO_RAD
                        ballData = self.ballDetect.updateBallData()
                        if ballData != {}:
                            break
                    self.motion.setAngles(Headnames, angle, 0.2)
                self.motion.moveTo(0.5, 0.0, _stepStatue)

        x = 0.0
        y = 0.0
        self.posture.goToPosture("StandInit", 0.5)
        time.sleep(1)
        theta = ballData["angle"]
        self.motion.setStiffnesses("Body", 1.0)
        time.sleep(1)
        self.motion.setMoveArmsEnabled(False, False)
        self.motion.moveTo(x, y, theta, _stepStatue)
        # self.tts.say("已转到面对球的方向")
        # 第一次转到机器人正对球的方向

        _stepStatue[3][1] = 0.2
        time.sleep(0.5)
        ballData = self.ballDetect.updateBallData(colorSpace="HSV", standState="standUp", saveFrameBin=True)
        x = ballData["disX"] - 0.2
        y = 0.0
        theta = 0.0
        # self.tts.say("距离红球还有" + str(x))
        self.motion.setStiffnesses("Body", 1.0)
        time.sleep(1)
        self.motion.setMoveArmsEnabled(False, False)
        self.motion.moveTo(x, y, theta, _stepStatue)
        self.motion.waitUntilMoveIsFinished()
        # 第二次走到距离球20cm的位置

        _stepStatue[3][1] = 0.6
        self.posture.goToPosture("StandInit", 0.5)
        # self.motion.waitUntilMoveIsFinshed()
        Headnames = "HeadPitch"
        timelist = 1.5
        angleLists = 30 * almath.TO_RAD
        isenabled = False  # False 表示相对角度
        self.motion.setStiffnesses("Body", 1.0)
        time.sleep(1)
        self.motion.angleInterpolation(Headnames, angleLists, timelist, isenabled)
        ballData = self.ballDetect.updateBallData(colorSpace="HSV", standState="standInit", saveFrameBin=True)
        x = 0.0
        y = 0.0
        theta = ballData["angle"]
        self.motion.setStiffnesses("Body", 1.0)
        time.sleep(1)
        self.motion.setMoveArmsEnabled(False, False)
        self.motion.moveTo(x, y, theta, _stepStatue)
        # self.tts.say("对准球的方向，修正完成")
        # 第三次机器人对准球,进行修正

        _stepStatue[3][1] = 0.5
        time.sleep(1.5)
        ballData = self.ballDetect.updateBallData(colorSpace="HSV", standState="standInit", saveFrameBin=True)
        # print (ballData)
        x = ballData["disX"] - 0.1
        y = 0.0
        theta = 0.0
        self.motion.setStiffnesses("Body", 1.0)
        time.sleep(1)
        self.motion.setMoveArmsEnabled(False, False)
        self.motion.moveTo(x, y, theta, _stepStatue)
        # self.tts.say("已走到离球10cm的位置")
        # 第四次机器人走到离球10cm的位置

        _stepStatue[3][1] = 0.6
        ballData = self.ballDetect.updateBallData(colorSpace="HSV", standState="standInit", saveFrameBin=True)
        dx = math.sqrt(ballData["disX"] * ballData["disX"] + ballData["disY"] * ballData["disY"])
        x = 0.0
        y = 0.0
        theta = ballData["angle"]
        self.motion.setStiffnesses("Body", 1.0)
        time.sleep(1)
        self.motion.moveTo(x, y, theta, _stepStatue)
        # self.tts.say("第二次修正完成")
        # 第五次机器人对准球进行修正

        ballData = self.ballDetect.updateBallData(colorSpace="HSV", standState="standInit", saveFrameBin=True)
        dx = math.sqrt(ballData["disX"] * ballData["disX"] + ballData["disY"] * ballData["disY"])
        x = ballData["disX"]
        y = ballData["disY"]
        theth = ballData["angle"]
        # self.tts.say("完成对球的最终定位")
        # 第六次对球进行最终定位

        Headnames = ["HeadPitch"]
        timelist = [0.5]
        angleLists = [0.0]
        self.motion.angleInterpolation(Headnames, angleLists, timelist, True)  # True 表示绝对角度
        for i in range(0, 20):
            time.sleep(1.5)
            landmark = LandMarkDetect(IP=robotIP)
            time.sleep(1.5)
            landmarkData = landmark.updateLandMarkData()
            if landmarkData != []:
                alpha = landmarkData[3]
                d1 = landmarkData[2]
                self.tts.say("我看到地标了")
                direction = 0
                break
            else:
                Headnames = ["HeadYaw"]
                timelist = [1.5]
                angleLists = [45 * almath.TO_RAD]
                self.motion.angleInterpolation(Headnames, angleLists, timelist, False)  # 相对角度
                time.sleep(3.5)
                landmarkData = landmark.updateLandMarkData()
                if landmarkData != []:
                    alpha = landmarkData[3]
                    d1 = landmarkData[2]
                    self.tts.say("我看到地标了")
                    direction = 1
                    break
                else:
                    Headnames = ["HeadYaw"]
                    timelist = [1.5]
                    angleLists = [-45 * almath.TO_RAD]
                    self.motion.angleInterpolation(Headnames, angleLists, timelist, True)
                    time.sleep(3.5)
                    landmarkData = landmark.updateLandMarkData()
                    if landmarkData != []:
                        alpha = landmarkData[3]
                        d1 = landmarkData[2]
                        self.tts.say("我看到地标了")
                        direction = -1
                        break
                    else:
                        Headnames = ["HeadYaw"]
                        timelist = [1.5]
                        angleLists = [0.0]
                        self.motion.angleInterpolation(Headnames, angleLists, timelist, True)
                        time.sleep(1.0)
                        self.motion.setStiffnesses("Body", 1.0)
                        time.sleep(1)
                        self.posture.goToPosture("StandInit", 0.5)
                        time.sleep(1)
                        self.motion.moveTo(0.1, 0.1, -math.pi / 2.0, _stepStatue)
                        time.sleep(3.5)
                        self.motion.waitUntilMoveIsFinished()
                        direction = 0
                        continue
        try:
            self.tts.say("距离地标的位置：" + str(landmarkData[2]))
        except:
            print str(landmarkData[2])
        print str(landmarkData)

        Headnames = ["HeadYaw"]
        timelist = [0.5]
        angleLists = [0.0]
        self.motion.angleInterpolation(Headnames, angleLists, timelist, True)  # True 表示绝对角度
        time.sleep(1.0)
        Headnames = ["HeadPitch"]
        timelist = [0.5]
        angleLists = [30 * almath.TO_RAD]
        self.motion.angleInterpolation(Headnames, angleLists, timelist, True)  # True 表示绝对角度
        time.sleep(1.5)
        ballData = self.ballDetect.updateBallData(colorSpace="HSV", standState="standInit", saveFrameBin=True)
        dx = math.sqrt(ballData["disX"] * ballData["disX"] + ballData["disY"] * ballData["disY"])
        x = ballData["disX"]
        y = ballData["disY"]
        time.sleep(5)
        Headnames = ["HeadPitch"]
        timelist = [0.5]
        angleLists = [0.0]
        self.motion.angleInterpolation(Headnames, angleLists, timelist, True)  # True 表示绝对角度
        theth = ballData["angle"]
        Headnames = ["HeadYaw"]
        timelist = [0.5]
        angleLists = [landmarkData[3]]
        self.motion.angleInterpolation(Headnames, angleLists, timelist, True)  # 将机器人的头对准mark

        time.sleep(1.5)
        # landmarkData = landmark.updateLandMarkData()
        if direction == 1:
            alpha += 45 * almath.TO_RAD
        elif direction == -1:
            alpha -= 45 * almath.TO_RAD
        else:
            alpha = alpha

        theta3 = abs(theth - alpha)  # 球与mark到机器人的夹角
        dball = dx  # 机器人与球之间的距离
        dmark = d1  # 机器人到mark的距离
        print ("dmark:", d1)
        d_ball_mark_square = dball ** 2 + dmark ** 2 - 2 * dball * dmark * math.cos(
            theta3)  # 余弦定理    (正弦定理OK)  求球到mark的距离
        print ("dmarksquare:", d_ball_mark_square)
        d_ball_mark = math.sqrt(d_ball_mark_square)
        theta_robot_ball_mark_data = (dball ** 2 + d_ball_mark_square - dmark ** 2) / (2 * dball * d_ball_mark)
        print ("ball_mark_robot_data", theta_robot_ball_mark_data)
        theta_robot_ball_mark = math.acos(theta_robot_ball_mark_data)  # 球到mark和机器人之间夹角的弧度
        theta_robot_mark_ball = math.pi - theta_robot_ball_mark - theta3  # 机器人到球和mark之间夹角的弧度
        print ("theta_ball_mark_robot", theta_robot_ball_mark)
        print ("theta_mark_ball_robot", theta_robot_mark_ball)
        print ("alpha", alpha)
        """
        if theth - alpha >= 0:
            if theta_ball_mark_robot >= math.pi / 2:
                theta = theta_ball_mark_robot - math.pi / 2
                x = dball * math.sin(theta_ball_mark_robot)
                y = dball * math.cos(theta_ball_mark_robot)
                y -= 0.10
            elif theta_ball_mark_robot < math.pi:
                theta = math.pi / 2 - theta_ball_mark_robot
                x = dball * math.sin(theta_ball_mark_robot)
                y = dball * math
        """
        if theth - alpha >= 0:  # 红球相对位置在mark右边
            if 0 <= theta_robot_ball_mark < 0.5 * math.pi:
                theta = 0.5 * math.pi - theta_robot_ball_mark
                dy = math.sqrt(((dball ** 2) * 2 - 2 * dball * dball * math.cos(theta)))
                x = -dy * math.cos(theta_robot_mark_ball)
                y = dy * math.sin(theta_robot_mark_ball)
                theta = theta
                print ("锐角theta_robot_ball_mark", theta_robot_ball_mark * almath.TO_DEG)
                self.tts.say("右边  夹角为锐角")
            else:
                theta = theta_robot_ball_mark - 0.5 * math.pi
                dy = (math.sqrt(((dball ** 2) * 2 - 2 * dball * dball * math.cos(theta))))
                thetatemp = (math.pi - theta) / 2.0
                x = -dy * math.sin(thetatemp)
                y = -dy * math.cos(thetatemp)
                theta = theta+10*almath.TO_RAD
                print ("钝角theta_robot_ba", theta_robot_ball_mark * almath.TO_DEG)
                self.tts.say("右边 夹角为钝角")
        else:
            if 0 <= theta_robot_ball_mark < 0.5 * math.pi:
                theta = 0.5 * math.pi - theta_robot_ball_mark
                dy = math.sqrt(((dball ** 2) * 2 - 2 * dball * dball * math.cos(theta)))
                x = -dy * math.cos(theta_robot_mark_ball)
                y = dy * math.sin(theta_robot_mark_ball)
                theta = -theta
                self.tts.say("左边  夹角为锐角")
            else:
                theta = theta_robot_ball_mark - 0.5 * math.pi
                dy = (math.sqrt(((dball ** 2) * 2 - 2 * dball * dball * math.cos(theta))))
                thetatemp = (math.pi - theta) / 2.0
                x = -dy * math.sin(thetatemp)
                y = dy * math.cos(thetatemp)
                y = y + 0.17
                # x = x + 0.03
                theta = -theta - 10 * almath.TO_RAD
                self.tts.say("左边 夹角为钝角")

        _stepStatue[4][1] = 0.025
        self.motion.setMoveArmsEnabled(False, False)
        self.motion.moveTo(0.0, 0.0, theta, _stepStatue)
        time.sleep(1)
        self.tts.say("转角完成")
        self.motion.waitUntilMoveIsFinished()
        self.motion.setStiffnesses("Body", 1.0)
        time.sleep(1)
        self.motion.setMoveArmsEnabled(False, False)
        self.motion.moveTo(x, 0.0, 0.0, _stepStatue)
        self.motion.waitUntilMoveIsFinished()
        self.motion.setMoveArmsEnabled(False, False)
        self.motion.moveTo(0.0, y, 0.0, _stepStatue)
        self.motion.waitUntilMoveIsFinished()

        # 再判定一次球的距离
        self.posture.goToPosture("StandInit", 0.5)
        time.sleep(1)
        Headnames = ["HeadPitch"]
        timelist = [0.5]
        angleLists = 30 * almath.TO_RAD
        self.motion.angleInterpolation(Headnames, angleLists, timelist, True)  # True 表示绝对角度
        ballData = self.ballDetect.updateBallData(colorSpace="HSV", standState="standInit", saveFrameBin=True)
        if ballData != []:
            x = ballData["disX"] - 0.1
            y = ballData["disY"]
            # theta = ballData["angle"]
        self.motion.setStiffnesses("Body", 1.0)
        time.sleep(1)
        # self.motion.setMoveArmsEnabled(False, False)
        # self.motion.moveTo(0.0, 0.0, theta, _stepStatue)
        # self.motion.waitUntilMoveIsFinished()
        self.motion.setMoveArmsEnabled(False, False)
        self.motion.moveTo(x, 0.0, 0.0, _stepStatue)
        self.motion.waitUntilMoveIsFinished()
        self.motion.setMoveArmsEnabled(False, False)
        self.motion.moveTo(0.0, y, 0.0, _stepStatue)
        self.motion.waitUntilMoveIsFinished()

        # slef.motion.moveTo(-)


if __name__ == "__main__":
    x = WalkToBall()
    x.start()
